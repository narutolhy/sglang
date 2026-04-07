# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.async_tp import (
    AsyncTpTensor,
    get_async_tp_min_tokens,
    log_async_tp_decision_once,
    should_use_async_tp_reduce_scatter,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_npu, make_layers
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
_is_npu = is_npu()

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
        use_async_tp: bool = False,
        is_scattered: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(
            x, use_fused_all_gather=(use_async_tp and is_scattered)
        )
        x = self.act_fn(gate_up)
        used_async_rs = False
        if use_async_tp:
            used_async_rs, _ = should_use_async_tp_reduce_scatter(
                x,
                self.down_proj.weight,
                self.down_proj.tp_size,
            )
        if used_async_rs:
            x, _ = self.down_proj(x, use_fused_reduce_scatter=True)
        else:
            x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)
        return x, used_async_rs


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        use_async_tp: bool = False,
        is_scattered: bool = False,
    ) -> torch.Tensor:
        num_tokens = positions.shape[0]

        # Async TP: fused AllGather+GEMM for qkv_proj when input is scattered
        if use_async_tp and is_scattered:
            qkv, _ = self.qkv_proj(hidden_states, use_fused_all_gather=True)
            # AG gathers shard_size*tp tokens which may exceed num_tokens
            # due to padding; truncate to M for attention (KV cache needs exact M)
            if qkv.shape[0] > num_tokens:
                qkv = qkv[:num_tokens]
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q, k = self.rotary_emb(positions, q, k)
        elif (
            not _is_npu
            or not hasattr(self.rotary_emb, "get_cos_sin_with_position")
            or forward_batch.forward_mode.is_extend()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        attn_output = self.attn(q, k, v, forward_batch)
        used_async_rs = False
        if use_async_tp:
            used_async_rs, _ = should_use_async_tp_reduce_scatter(
                attn_output,
                self.o_proj.weight,
                self.o_proj.tp_size,
            )
        if used_async_rs:
            # Pad attn_output to nearest multiple of tp_size for GEMM+RS
            tp_size = get_tensor_model_parallel_world_size()
            pad_size = (tp_size - attn_output.shape[0] % tp_size) % tp_size
            if pad_size > 0:
                attn_output = torch.nn.functional.pad(
                    attn_output, (0, 0, 0, pad_size)
                )
            output, _ = self.o_proj(attn_output, use_fused_reduce_scatter=True)
        else:
            output, _ = self.o_proj(attn_output)
        return output, used_async_rs


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Async TP config
        server_args = get_global_server_args()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_async_tp = (
            server_args is not None
            and server_args.enable_async_tp
            and self.tp_size > 1
        )
        # Compile-compatible SP mode: uses plain RS/AG custom_ops
        # instead of fused symm_mem ops. Works in both eager and compile modes,
        # but designed to be torch.compile-compatible (no c10d._resolve_process_group).
        self.use_compile_sp = self.use_async_tp

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_compile_sp:
            return self._forward_compile_sp(
                positions, hidden_states, forward_batch, residual
            )
        if self.use_async_tp:
            return self._forward_async_tp(
                positions, hidden_states, forward_batch, residual
            )

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, _ = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, _ = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_compile_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compile-compatible sequence parallelism.

        Uses plain RS/AG (registered custom_ops, NOT split_ops) so the
        entire layer stays inside one compiled subgraph / CUDA graph.
        No fused symm_mem ops, no Python stream management.

        Decode falls back to standard path (no SP for M=1).
        """
        num_tokens = positions.shape[0]

        # Decode: use standard path (no SP benefit for tiny GEMM)
        if num_tokens == 0 or forward_batch.forward_mode.is_decode():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual
                )
            hidden_states, _ = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states, _ = self.mlp(hidden_states)
            return hidden_states, residual

        # Detect scattered state from previous layer
        is_scattered = getattr(
            hidden_states, "_async_tp_scattered", False
        ) or hidden_states.shape[0] != num_tokens

        # LayerNorm + residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # --- Self Attention ---
        # qkv_proj: AG + GEMM if scattered
        if is_scattered:
            qkv, _ = self.self_attn.qkv_proj.forward_ag_gemm(hidden_states)
            # AG gathers to padded_m; truncate to M for attention
            if qkv.shape[0] > num_tokens:
                qkv = qkv[:num_tokens]
            q, k, v = qkv.split(
                [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
                dim=-1,
            )
            q, k = self.self_attn.rotary_emb(positions, q, k)
        else:
            q, k, v = self.self_attn.forward_prepare_native(positions, hidden_states)

        attn_output = self.self_attn.attn(q, k, v, forward_batch)

        # o_proj: GEMM + RS
        # Pad to nearest multiple of tp_size for RS
        tp_size = self.tp_size
        pad_size = (tp_size - attn_output.shape[0] % tp_size) % tp_size
        if pad_size > 0:
            attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, pad_size))
        hidden_states = self.self_attn.o_proj.forward_gemm_rs(attn_output)

        # Scatter residual to match
        if not is_scattered:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = hidden_states.shape[0]
            if pad_size > 0:
                residual = torch.nn.functional.pad(residual, (0, 0, 0, pad_size))
            residual = residual.narrow(0, tp_rank * shard_size, shard_size)

        # --- MLP ---
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        # gate_up_proj: AG + GEMM
        gate_up, _ = self.mlp.gate_up_proj.forward_ag_gemm(hidden_states)
        x = self.mlp.act_fn(gate_up)
        # down_proj: GEMM + RS
        hidden_states = self.mlp.down_proj.forward_gemm_rs(x)

        # Mark as scattered for next layer
        hidden_states._async_tp_scattered = True
        return hidden_states, residual

    def _forward_async_tp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with async TP: GEMM+ReduceScatter / AllGather+GEMM overlap.

        Padding is handled inside self_attn.forward() and self.mlp.forward():
        - AG+GEMM may gather to padded_m > M; qkv is truncated to M for attention
        - attn_output is padded to nearest multiple of tp_size before GEMM+RS
        - MLP AG+GEMM and GEMM+RS naturally work on padded sizes
        """
        num_tokens = positions.shape[0]
        hidden_state_meta = AsyncTpTensor.from_tensor(
            hidden_states,
            logical_num_tokens=num_tokens,
            tp_size=self.tp_size,
        )
        is_scattered = hidden_state_meta.is_scattered()

        decode_min_tokens = get_async_tp_min_tokens("decode")
        decode_layer_min_tokens = max(
            decode_min_tokens,
            get_async_tp_min_tokens("ag"),
            get_async_tp_min_tokens("rs"),
        )
        use_decode_async_tp = (
            forward_batch.forward_mode.is_decode()
            and num_tokens >= decode_layer_min_tokens
        )
        if (
            forward_batch.forward_mode.is_decode()
            and not use_decode_async_tp
            and decode_layer_min_tokens > 0
        ):
            log_async_tp_decision_once(
                f"llama_decode_fallback:{self.self_attn.attn.layer_id}:{num_tokens}:{decode_layer_min_tokens}",
                f"{self.self_attn.attn.layer_id=}: fallback to standard decode path because num_tokens={num_tokens}<decode_layer_min_tokens={decode_layer_min_tokens}",
            )

        if num_tokens == 0 or (
            forward_batch.forward_mode.is_decode() and not use_decode_async_tp
        ):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual
                )
            hidden_states, _ = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states, _ = self.mlp(hidden_states)
            return hidden_states, residual

        # Self Attention
        if residual is None:
            residual = hidden_states
            residual_meta = hidden_state_meta.with_tensor(residual)
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            residual_meta = AsyncTpTensor.from_tensor(
                residual,
                logical_num_tokens=num_tokens,
                tp_size=self.tp_size,
            )

        hidden_states, attn_used_async_rs = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            use_async_tp=True,
            is_scattered=is_scattered,
        )
        if attn_used_async_rs:
            hidden_state_meta = hidden_state_meta.mark_scattered(hidden_states)
        else:
            hidden_state_meta = hidden_state_meta.with_tensor(hidden_states)
        hidden_states = hidden_state_meta.attach()

        # Scatter residual to match if not already scattered
        if not is_scattered:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = hidden_states.shape[0]
            # Pad residual if M is not divisible by tp_size
            pad_size = shard_size * self.tp_size - num_tokens
            if pad_size > 0:
                residual = torch.nn.functional.pad(
                    residual, (0, 0, 0, pad_size)
                )
            residual = residual.narrow(0, tp_rank * shard_size, shard_size)
            residual_meta = residual_meta.mark_scattered(
                residual,
                padded_num_tokens=hidden_state_meta.padded_num_tokens,
            )
        else:
            residual_meta = residual_meta.with_tensor(residual)
        residual = residual_meta.attach()

        # Fully Connected (both hidden_states and residual are scattered)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_state_meta = hidden_state_meta.with_tensor(hidden_states)
        residual_meta = residual_meta.with_tensor(residual)
        hidden_states = hidden_state_meta.attach()
        residual = residual_meta.attach()
        hidden_states, mlp_used_async_rs = self.mlp(
            hidden_states, use_async_tp=True, is_scattered=True
        )
        if mlp_used_async_rs:
            hidden_states = hidden_state_meta.mark_scattered(
                hidden_states,
                padded_num_tokens=hidden_state_meta.padded_num_tokens,
            ).attach()
        else:
            hidden_states = hidden_state_meta.with_tensor(hidden_states).attach()
        residual = residual_meta.attach()
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []

    def _finalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        hidden_state_meta = AsyncTpTensor.from_tensor(
            hidden_states,
            logical_num_tokens=num_tokens,
            tp_size=get_tensor_model_parallel_world_size(),
        )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_state_meta = hidden_state_meta.with_tensor(hidden_states)

        if not hidden_state_meta.is_scattered():
            return hidden_states

        tp_group = get_tp_group()
        gathered_size = hidden_state_meta.padded_num_tokens
        full_hidden = torch.empty(
            gathered_size,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        tp_group.all_gather_into_tensor(full_hidden, hidden_states)
        return full_hidden[:num_tokens]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            hidden_states = self._finalize_hidden_states(
                hidden_states,
                residual,
                positions.shape[0],
            )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the model."""
        return self.embed_tokens


class LlamaForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        self.capture_aux_hidden_states = False

    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return LlamaModel(config, quant_config=quant_config, prefix=prefix)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            hidden_states = self.model._finalize_hidden_states(
                forward_batch.hidden_states,
                forward_batch.residual,
                positions.shape[0],
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Phi3ForCausalLM(LlamaForCausalLM):
    pass


class InternLM3ForCausalLM(LlamaForCausalLM):
    pass


class IQuestCoderForCausalLM(LlamaForCausalLM):
    pass


EntryClass = [
    LlamaForCausalLM,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
    IQuestCoderForCausalLM,
]
