"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
    kv_to_page_indices,
    poll_and_all_reduce,
    prepare_abort,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    KVCache,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.utils import get_int_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

_STAGING_DEBUG = os.getenv("SGLANG_STAGING_DEBUG", "0") == "1"

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

CLIP_MAX_NEW_TOKEN = get_int_env_var("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", 4096)


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (size + pre_alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(size + pre_alloc_size))

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        chunked = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(chunked) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in chunked
        ), "request has req_pool_idx but is not chunked"

        need_size = len(reqs) - len(chunked)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: "Req"):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: "Mamba2CacheParams",
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        mamba_size: int = None,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )
        self.mamba_ping_pong_track_buffer_size = (
            2 if speculative_num_draft_tokens is None else 1
        )
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        effective_mamba_size = (
            mamba_size if mamba_size is not None else size
        ) + pre_alloc_size
        self._init_mamba_pool(
            size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def clear(self):
        self.free_slots = list(range(self.size + self.pre_alloc_size))
        self.mamba_pool.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1

    @property
    def seqlen(self) -> int:
        return self.req.seqlen


class DecodePreallocQueue:
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        prefill_pp_size: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache  # this is always a chunk cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.prefill_pp_size = prefill_pp_size
        self.pp_rank = pp_rank
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[Req] = []
        self.prefill_pp_size = prefill_pp_size
        self.kv_manager = self._init_kv_manager()
        self.transfer_queue._init_staging_ctx()

        if self.scheduler.tp_worker.is_hybrid_swa:
            # FIXME: current SWA allocation allocate full kv cache size in prefill
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_attention_tp_size()
        kv_args.engine_rank = self.tp_rank

        kv_args.decode_tp_size = attn_tp_size
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.dp_rank
        kv_args.prefill_pp_size = self.prefill_pp_size
        kv_pool_for_heads = self.token_to_kv_pool
        if hasattr(kv_pool_for_heads, "full_kv_pool"):
            kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
        per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
        if per_rank_kv_heads > 0:
            kv_args.kv_head_num = per_rank_kv_heads
            kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        if hasattr(self.token_to_kv_pool, "get_state_buf_infos"):
            state_data_ptrs, state_data_lens, state_item_lens = (
                self.token_to_kv_pool.get_state_buf_infos()
            )
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            if isinstance(self.token_to_kv_pool, SWAKVPool):
                kv_args.state_type = "swa"
            elif isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                kv_args.state_type = "mamba"
                # Get state dimension info for cross-TP slice transfer
                if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                    kv_args.state_dim_per_tensor = (
                        self.token_to_kv_pool.get_state_dim_per_tensor()
                    )
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                kv_args.state_type = "nsa"
            else:
                kv_args.state_type = "none"
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs for staging buffer scatter on decode side
        if (
            hasattr(kv_manager, "enable_staging")
            and kv_manager.enable_staging
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        # Both will be set from actual prefill info in
        # _resolve_pending_reqs after ensure_parallel_info succeeds.
        kv_manager.prefill_attn_tp_size = 0
        kv_manager.prefill_chunked_prefill_size = 0
        self.scheduler._decode_kv_manager = kv_manager
        return kv_manager

    def add(self, req: Req, is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            if prefill_dp_rank is None:
                self.pending_reqs.append(req)
                return
            self._create_receiver_and_enqueue(req, prefill_dp_rank)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if _is_fake_transfer(req, self.scheduler.server_args):
            return 0

        bootstrap_addr = f"{req.bootstrap_host}:{req.bootstrap_port}"

        prefill_info = self.kv_manager.prefill_info_table.get(bootstrap_addr)
        if prefill_info is None:
            return None

        if prefill_info.dp_size == 1:
            return 0

        if prefill_info.follow_bootstrap_room:
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(self, req: Req, prefill_dp_rank: int) -> None:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=prefill_dp_rank,
        )

        self.queue.append(
            DecodeRequest(req=req, kv_receiver=kv_receiver, waiting_for_input=False)
        )

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.stream_output([req], req.return_logprob)
            return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        allocatable_tokens = self._allocatable_tokens(count_retracted=False)

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            required_tokens_for_request = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                + self.num_reserved_decode_tokens
            )
            if required_tokens_for_request > allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            allocatable_tokens -= required_tokens_for_request

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self, rids_to_check: Optional[List[str]] = None
    ) -> None:
        should_poll = len(self.queue) > 0 and not all(
            d.waiting_for_input for d in self.queue
        )
        n = len(self.queue)
        guard = torch.tensor([int(should_poll), n, -n], dtype=torch.int64, device="cpu")
        dist.all_reduce(guard, op=dist.ReduceOp.MIN, group=self.gloo_group)
        if guard[0].item() == 0 or guard[1].item() != -guard[2].item():
            return

        polls = poll_and_all_reduce(
            [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
        )

        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and create receivers."""
        if not self.pending_reqs:
            return

        bootstrap_addr = f"{self.pending_reqs[0].bootstrap_host}:{self.pending_reqs[0].bootstrap_port}"

        # If a request is following the bootstrap room,
        # we need get the prefill info before resolving the prefill_dp_ranks
        # which is a conflict with the lazy resolve logic in CommonKVReceiver,
        # so we need to ensure the parallel info before resolving it.
        if not self.kv_manager.ensure_parallel_info(bootstrap_addr):
            return

        if self.kv_manager.prefill_attn_tp_size == 0:
            prefill_info = self.kv_manager.prefill_info_table.get(bootstrap_addr)
            if prefill_info is not None:
                self.kv_manager.prefill_attn_tp_size = prefill_info.attn_tp_size
                server_args = self.scheduler.server_args
                decode_dp = self.dp_size or 1
                decode_cps = server_args.chunked_prefill_size
                if getattr(server_args, "enable_dp_attention", False) and decode_dp > 1:
                    original_cps = decode_cps * decode_dp
                else:
                    original_cps = decode_cps
                prefill_dp = prefill_info.dp_size or 1
                self.kv_manager.prefill_chunked_prefill_size = (
                    original_cps // prefill_dp
                )
                self.transfer_queue._init_staging_ctx()

        resolved = []
        need_query = []
        for req in self.pending_reqs:
            # NOTE: we need resolve it again because we may ensure the parallel info here
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            if prefill_dp_rank is not None:
                resolved.append((req, prefill_dp_rank))
            else:
                need_query.append(req)

        if need_query:
            from sglang.srt.disaggregation.common.conn import CommonKVReceiver

            rooms = [req.bootstrap_room for req in need_query]
            room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                bootstrap_addr, rooms
            )
            remaining = []
            for req in need_query:
                room_key = str(req.bootstrap_room)
                if room_key in room_to_rank:
                    resolved.append((req, int(room_to_rank[room_key])))
                else:
                    remaining.append(req)
            self.pending_reqs = remaining
        else:
            self.pending_reqs = []

        for req, prefill_dp_rank in resolved:
            self._create_receiver_and_enqueue(req, prefill_dp_rank)

    def pop_preallocated(
        self, rids_to_check: Optional[List[str]] = None
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """Pop the preallocated requests from the pending queue (FIFO)."""
        self._resolve_pending_reqs()
        self._update_handshake_waiters(rids_to_check)

        failed_reqs = []
        preallocated_reqs = []
        indices_to_remove = set()

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        allocatable_tokens = self._allocatable_tokens(
            retractable_tokens=retractable_tokens, count_retracted=True
        )
        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                failed_reqs.append(decode_req)
                indices_to_remove.add(i)

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if i in indices_to_remove:
                continue

            if not decode_req.waiting_for_input:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                break

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = len(decode_req.req.origin_input_ids)
            required_tokens_for_request = (
                origin_input_len + self.num_reserved_decode_tokens
            )

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > allocatable_tokens
            ):
                break
            if required_tokens_for_request > allocatable_tokens:
                break

            allocatable_tokens -= required_tokens_for_request
            self._pre_alloc(decode_req.req)

            kv_indices = (
                self.req_to_token_pool.req_to_token[decode_req.req.req_pool_idx][
                    : len(decode_req.req.origin_input_ids)
                ]
                .cpu()
                .numpy()
            )
            page_size = self.token_to_kv_pool_allocator.page_size

            # Prepare extra pool indices for hybrid models
            if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
                # Mamba hybrid model: single mamba state index
                state_indices = [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
            elif isinstance(self.token_to_kv_pool, SWAKVPool):
                # SWA hybrid model: send decode-side SWA window indices
                seq_len = len(decode_req.req.origin_input_ids)
                window_size = self.scheduler.sliding_window_size

                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]

                # Translate to SWA pool indices
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                state_indices = window_kv_indices_swa.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            elif isinstance(self.token_to_kv_pool, NSATokenToKVPool):
                seq_len = len(decode_req.req.origin_input_ids)
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                state_indices = kv_indices_full.cpu().numpy()
                state_indices = kv_to_page_indices(state_indices, page_size)
            else:
                state_indices = None

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            assert decode_req.metadata_buffer_index is not None
            page_indices = kv_to_page_indices(kv_indices, page_size)
            decode_req.kv_receiver.init(
                page_indices, decode_req.metadata_buffer_index, state_indices
            )
            preallocated_reqs.append(decode_req)
            indices_to_remove.add(i)
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return preallocated_reqs, failed_reqs

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            len(decode_req.req.fill_ids) for decode_req in self.transfer_queue.queue
        )

    def _allocatable_tokens(
        self, retractable_tokens: Optional[int] = None, count_retracted: bool = True
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        available_size = self.token_to_kv_pool_allocator.available_size()
        allocatable_tokens = available_size - max(
            # preserve some space for future decode
            self.num_reserved_decode_tokens
            * (
                len(self.scheduler.running_batch.reqs)
                + len(self.transfer_queue.queue)
                + len(self.scheduler.waiting_queue)
            ),
            # make sure each request can finish if reach max_tokens with all other requests retracted
            need_space_for_single_req,
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            allocatable_tokens -= sum(
                [
                    len(req.origin_input_ids)
                    + len(req.output_ids)
                    + self.num_reserved_decode_tokens
                    for req in self.retracted_queue
                ]
            )
        return allocatable_tokens

    def _pre_alloc(self, req: Req) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool"""
        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        # Alloc all tokens for the prebuilt req (except for the reserved input token for decoding)
        fill_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        req.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        if self.token_to_kv_pool_allocator.page_size == 1:
            kv_loc = self.token_to_kv_pool_allocator.alloc(fill_len)
        else:
            device = self.token_to_kv_pool_allocator.device
            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
                extend_num_tokens=fill_len,
            )

        assert (
            kv_loc is not None
        ), "KV cache is full! There is a bug in memory estimation."

        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)

        # populate metadata
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.set_extend_input_len(len(req.fill_ids))

        return kv_loc


class DecodeTransferQueue:
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.staging_ctx = None

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest) -> bool:
        """
        Returns:
            True if the request should be removed from the queue (success or corruption)
            False if metadata not ready yet (keep in queue for next poll)
        """
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption.
        # Apply same int64 mask as set_buf() for consistent comparison.
        actual_room = output_bootstrap_room[0].item()
        raw_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )
        expected_room = raw_room & 0x7FFFFFFFFFFFFFFF

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Case 1: Metadata not ready yet (actual_room == 0)
            # Keep request in queue and wait for next poll
            return False
        elif actual_room != expected_room:
            # Case 2: Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return True

        # Case 3: Success - commit the transfer
        decode_req.req.output_ids.append(output_id[0].item())
        decode_req.req.cached_tokens = cached_tokens[0].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states

        if decode_req.req.return_logprob:
            decode_req.req.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.output_top_logprobs_val.append(
                output_top_logprobs_val[: decode_req.req.top_logprobs_num].tolist()
            )
            decode_req.req.output_top_logprobs_idx.append(
                output_top_logprobs_idx[: decode_req.req.top_logprobs_num].tolist()
            )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return True

    def _init_staging_ctx(self):
        """Initialize staging context after kv_manager is available."""
        kv_manager = getattr(self.scheduler, "_decode_kv_manager", None)
        if kv_manager is None:
            return
        staging_allocator = getattr(kv_manager, "staging_allocator", None)
        if staging_allocator is None:
            return
        kv_buffer_info = getattr(kv_manager, "kv_buffer_tensors", None)
        if kv_buffer_info is None:
            return
        prefill_tp = getattr(kv_manager, "prefill_attn_tp_size", 0)
        decode_tp = kv_manager.attn_tp_size
        if prefill_tp == 0 or prefill_tp == decode_tp:
            return
        from sglang.srt.disaggregation.common.staging import resolve_total_kv_heads

        total_kv_heads = resolve_total_kv_heads(
            kv_manager.kv_args,
            decode_tp,
            kv_buffer_tensors=kv_buffer_info,
        )
        self.staging_ctx = (
            kv_manager,
            staging_allocator,
            kv_buffer_info,
            prefill_tp,
            decode_tp,
            total_kv_heads,
        )

    def _e2e_verify_staging(
        self,
        staging_view,
        k_buffers,
        v_buffers,
        page_idx_tensor,
        page_size,
        prefill_tp,
        decode_tp,
        dst_tp_rank,
        total_kv_heads,
        staging_offset,
        num_pages,
        decode_req,
    ):
        """Compare staging buffer vs KV cache (written by slice ground truth).

        Only runs when SGLANG_STG_TRACE=1 and prefill double-writes.
        """
        from sglang.srt.disaggregation.common.staging import compute_head_slice_params

        kv_manager = getattr(self.scheduler, "_decode_kv_manager", None)
        engine_rank = (
            getattr(getattr(kv_manager, "kv_args", None), "engine_rank", -1)
            if kv_manager
            else -1
        )

        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()
        num_tokens = page_idx_tensor.shape[0] * page_size

        if page_size > 1:
            offsets = torch.arange(page_size, device=page_idx_tensor.device)
            token_indices = (
                page_idx_tensor.unsqueeze(1) * page_size + offsets
            ).reshape(-1)
        else:
            token_indices = page_idx_tensor

        num_writers = prefill_tp // max(1, decode_tp) if prefill_tp > decode_tp else 1

        receiver = getattr(decode_req, "kv_receiver", None)
        chunk_infos = getattr(receiver, "chunk_staging_infos", [])
        seq_len = len(decode_req.req.origin_input_ids)

        common = (
            f"rid={decode_req.req.rid} room={decode_req.req.bootstrap_room} "
            f"engine_rank={engine_rank} attn_tp_rank={dst_tp_rank} "
            f"prefill_tp={prefill_tp} decode_tp={decode_tp} dp_size={self.scheduler.server_args.dp_size} "
            f"kv_heads={total_kv_heads} num_layers={num_layers} "
            f"tokens={num_tokens} pages={num_pages} page_size={page_size} "
            f"seq_len={seq_len} writers={num_writers} "
            f"stg_offset={staging_offset} num_chunks={len(chunk_infos)} "
            f"head_dim={head_dim} dtype_size={dtype_size}"
        )

        torch.cuda.synchronize()
        mismatch_layers = []
        all_layer_sums = []

        for writer_rank in range(num_writers):
            _, nh, dhs, _ = compute_head_slice_params(
                prefill_tp, decode_tp, writer_rank, dst_tp_rank, total_kv_heads
            )
            plb = num_tokens * nh * head_dim * dtype_size
            prb = plb * num_layers * 2
            rb = writer_rank * prb
            off = rb
            for lid in range(num_layers):
                stg = staging_view[off : off + plb]
                kv_slice = k_buffers[lid][token_indices, dhs : dhs + nh, :]
                kv_bytes = kv_slice.contiguous().view(-1).view(torch.uint8)
                ss = int(stg.sum(dtype=torch.int64).item())
                ks = int(kv_bytes.sum(dtype=torch.int64).item())
                matched = torch.equal(stg, kv_bytes)
                all_layer_sums.append(
                    f"K{lid}:w{writer_rank}:stg={ss}:kv={ks}:{'OK' if matched else 'FAIL'}"
                )
                if not matched:
                    diff = (stg != kv_bytes).sum().item()
                    stg_zero = int((stg == 0).sum().item())
                    kv_zero = int((kv_bytes == 0).sum().item())
                    mismatch_layers.append(
                        f"K{lid} w{writer_rank} head={dhs} diff={diff}/{kv_bytes.numel()} "
                        f"stg_sum={ss} kv_sum={ks} stg_zeros={stg_zero} kv_zeros={kv_zero}"
                    )
                off += plb
            for lid in range(num_layers):
                stg = staging_view[off : off + plb]
                kv_slice = v_buffers[lid][token_indices, dhs : dhs + nh, :]
                kv_bytes = kv_slice.contiguous().view(-1).view(torch.uint8)
                ss = int(stg.sum(dtype=torch.int64).item())
                ks = int(kv_bytes.sum(dtype=torch.int64).item())
                matched = torch.equal(stg, kv_bytes)
                all_layer_sums.append(
                    f"V{lid}:w{writer_rank}:stg={ss}:kv={ks}:{'OK' if matched else 'FAIL'}"
                )
                if not matched:
                    diff = (stg != kv_bytes).sum().item()
                    stg_zero = int((stg == 0).sum().item())
                    kv_zero = int((kv_bytes == 0).sum().item())
                    mismatch_layers.append(
                        f"V{lid} w{writer_rank} head={dhs} diff={diff}/{kv_bytes.numel()} "
                        f"stg_sum={ss} kv_sum={ks} stg_zeros={stg_zero} kv_zeros={kv_zero}"
                    )
                off += plb

        if mismatch_layers:
            logger.error(
                f"[E2E MISMATCH] {len(mismatch_layers)} layers | {common}\n"
                + "\n".join(f"  {m}" for m in mismatch_layers)
            )
            logger.error(f"[E2E SUMS] {common}\n" + " | ".join(all_layer_sums))
        else:
            logger.info(f"[E2E OK] {common}")

    def _scatter_staging_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: DecodeRequest,
    ) -> bool:
        """Submit scatter kernels for a staging region to scatter_stream.

        Works for both per-chunk and full-request scatter.
        Returns True if scatter was submitted, False otherwise.
        """
        ctx = self.staging_ctx
        if ctx is None:
            return False
        (
            kv_manager,
            staging_allocator,
            kv_buffer_info,
            prefill_tp,
            decode_tp,
            total_kv_heads,
        ) = ctx

        from sglang.srt.disaggregation.common.staging import scatter_staging_to_kv

        k_buffers = kv_buffer_info["k_buffers"]
        v_buffers = kv_buffer_info["v_buffers"]
        page_size = kv_buffer_info["page_size"]
        dst_tp_rank = kv_manager.kv_args.engine_rank % decode_tp

        req_pool_idx = decode_req.req.req_pool_idx
        token_start = page_start * page_size
        token_end = token_start + num_pages * page_size
        kv_indices = self.scheduler.req_to_token_pool.req_to_token[
            req_pool_idx, token_start:token_end
        ]
        if page_size > 1:
            from sglang.srt.disaggregation.utils import kv_to_page_indices

            kv_indices = kv_to_page_indices(kv_indices.cpu().numpy(), page_size)
            page_idx_tensor = torch.from_numpy(kv_indices).to(k_buffers[0].device)
        else:
            page_idx_tensor = kv_indices.to(k_buffers[0].device)

        staging_view = staging_allocator.buffer.buffer[staging_offset:]

        if not hasattr(staging_allocator, "_scatter_stream"):
            staging_allocator._scatter_stream = torch.cuda.Stream()

        staging_view[0].item()

        if os.getenv("SGLANG_STG_TRACE", "0") == "1":
            self._e2e_verify_staging(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                decode_tp,
                dst_tp_rank,
                total_kv_heads,
                staging_offset,
                num_pages,
                decode_req,
            )

        with torch.cuda.stream(staging_allocator._scatter_stream):
            scatter_staging_to_kv(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                decode_tp,
                dst_tp_rank,
                total_kv_heads,
            )

        if os.getenv("SGLANG_STG_TRACE", "0") == "1":
            staging_allocator._scatter_stream.synchronize()
            self._post_scatter_verify(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                decode_tp,
                dst_tp_rank,
                total_kv_heads,
                num_pages,
                decode_req,
            )

        return True

    def _post_scatter_verify(
        self,
        staging_view,
        k_buffers,
        v_buffers,
        page_idx_tensor,
        page_size,
        prefill_tp,
        decode_tp,
        dst_tp_rank,
        total_kv_heads,
        num_pages,
        decode_req,
    ):
        """Compare KV cache AFTER scatter vs staging buffer.

        Catches bugs in scatter_staging_to_kv (e.g. wrong head placement).
        """
        from sglang.srt.disaggregation.common.staging import compute_head_slice_params

        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()
        num_tokens = page_idx_tensor.shape[0] * page_size

        if page_size > 1:
            offsets = torch.arange(page_size, device=page_idx_tensor.device)
            token_indices = (
                page_idx_tensor.unsqueeze(1) * page_size + offsets
            ).reshape(-1)
        else:
            token_indices = page_idx_tensor

        num_writers = prefill_tp // max(1, decode_tp) if prefill_tp > decode_tp else 1

        rid = getattr(decode_req.req, "rid", "?")
        room = getattr(decode_req.req, "bootstrap_room", "?")

        mismatch_layers = []
        for writer_rank in range(num_writers):
            _, nh, dhs, _ = compute_head_slice_params(
                prefill_tp, decode_tp, writer_rank, dst_tp_rank, total_kv_heads
            )
            plb = num_tokens * nh * head_dim * dtype_size
            prb = plb * num_layers * 2
            off = writer_rank * prb
            for lid in range(num_layers):
                stg = staging_view[off : off + plb]
                kv_slice = k_buffers[lid][token_indices, dhs : dhs + nh, :]
                kv_bytes = kv_slice.contiguous().view(-1).view(torch.uint8)
                if not torch.equal(stg, kv_bytes):
                    diff = (stg != kv_bytes).sum().item()
                    mismatch_layers.append(
                        f"K{lid} w{writer_rank} head={dhs} diff={diff}/{kv_bytes.numel()}"
                    )
                off += plb
            for lid in range(num_layers):
                stg = staging_view[off : off + plb]
                kv_slice = v_buffers[lid][token_indices, dhs : dhs + nh, :]
                kv_bytes = kv_slice.contiguous().view(-1).view(torch.uint8)
                if not torch.equal(stg, kv_bytes):
                    diff = (stg != kv_bytes).sum().item()
                    mismatch_layers.append(
                        f"V{lid} w{writer_rank} head={dhs} diff={diff}/{kv_bytes.numel()}"
                    )
                off += plb

        if mismatch_layers:
            logger.error(
                f"[POST-SCATTER MISMATCH] {len(mismatch_layers)} layers | "
                f"rid={rid} room={room} "
                f"prefill_tp={prefill_tp} decode_tp={decode_tp} "
                f"dst_tp_rank={dst_tp_rank} kv_heads={total_kv_heads} "
                f"tokens={num_tokens} pages={num_pages}\n"
                + "\n".join(f"  {m}" for m in mismatch_layers)
            )
        else:
            if _STAGING_DEBUG:
                logger.info(
                    "[POST-SCATTER OK] rid=%s room=%s "
                    "prefill_tp=%s decode_tp=%s tokens=%d pages=%d",
                    rid,
                    room,
                    prefill_tp,
                    decode_tp,
                    num_tokens,
                    num_pages,
                )

    def _submit_scatter_staging(self, decode_req: DecodeRequest) -> int:
        """Submit scatter for the last chunk (triggered by KVPoll.Success).

        Returns the alloc_id (>= 0) if scatter was submitted, or -1.
        """
        if self.staging_ctx is None:
            if _STAGING_DEBUG:
                logger.warning(
                    "[LAST-SCATTER-SKIP] decode_tp=%d rid=%s room=%s reason=no_staging_ctx",
                    self.tp_rank,
                    decode_req.req.rid,
                    decode_req.req.bootstrap_room,
                )
            return -1

        kv_manager = self.staging_ctx[0]
        receiver = decode_req.kv_receiver
        chunk_infos = getattr(receiver, "chunk_staging_infos", [])
        if not chunk_infos:
            if _STAGING_DEBUG:
                logger.warning(
                    "[LAST-SCATTER-SKIP] decode_tp=%d rid=%s room=%s session=%s reason=empty_chunk_infos",
                    self.tp_rank,
                    decode_req.req.rid,
                    decode_req.req.bootstrap_room,
                    getattr(receiver, "session_id", ""),
                )
            return -1

        last_info = chunk_infos[-1]
        alloc_id, staging_offset, _, _ = last_info
        if staging_offset < 0 or alloc_id < 0:
            if _STAGING_DEBUG:
                logger.warning(
                    "[LAST-SCATTER-SKIP] decode_tp=%d rid=%s room=%s session=%s "
                    "reason=invalid_last_info alloc_id=%d staging_offset=%d chunk_infos=%s",
                    self.tp_rank,
                    decode_req.req.rid,
                    decode_req.req.bootstrap_room,
                    getattr(receiver, "session_id", ""),
                    alloc_id,
                    staging_offset,
                    chunk_infos,
                )
            return -1

        seq_len = len(decode_req.req.origin_input_ids)
        allocator = getattr(self.scheduler, "token_to_kv_pool_allocator", None)
        ps = allocator.page_size if allocator else 1
        total_pages = (seq_len + ps - 1) // ps

        n = len(chunk_infos)
        prefill_cps = (
            getattr(kv_manager, "prefill_chunked_prefill_size", None)
            or self.scheduler.server_args.chunked_prefill_size
            or 8192
        )
        chunk_pages = max(1, prefill_cps // ps)
        page_start = chunk_pages * (n - 1)
        last_num_pages = total_pages - page_start

        if _STAGING_DEBUG:
            logger.info(
                "[LAST-SCATTER-SUBMIT] decode_tp=%d rid=%s room=%s session=%s "
                "alloc_id=%d staging_offset=%d total_pages=%d num_chunks=%d "
                "chunk_pages=%d page_start=%d last_num_pages=%d",
                self.tp_rank,
                decode_req.req.rid,
                decode_req.req.bootstrap_room,
                getattr(receiver, "session_id", ""),
                alloc_id,
                staging_offset,
                total_pages,
                n,
                chunk_pages,
                page_start,
                last_num_pages,
            )
        ok = self._scatter_staging_region(
            staging_offset, page_start, last_num_pages, decode_req
        )
        if _STAGING_DEBUG:
            logger.info(
                "[LAST-SCATTER-RESULT] decode_tp=%d rid=%s room=%s session=%s "
                "alloc_id=%d ok=%s",
                self.tp_rank,
                decode_req.req.rid,
                decode_req.req.bootstrap_room,
                getattr(receiver, "session_id", ""),
                alloc_id,
                ok,
            )
        return alloc_id if ok else -1

    def _free_and_send_watermark(self, alloc_id: int, decode_req: DecodeRequest):
        """Free a staging allocation and send watermark to prefill."""
        ctx = self.staging_ctx
        if ctx is None:
            return
        _, staging_allocator, _, _, _, _ = ctx
        if _STAGING_DEBUG:
            _pre_wm = staging_allocator.get_watermark()
            _num_allocs_before = len(staging_allocator.allocations)
        staging_allocator.free(alloc_id)
        post_wm = staging_allocator.get_watermark()
        if _STAGING_DEBUG:
            logger.info(
                "[WM-FREE] decode_tp=%d alloc_id=%d room=%s "
                "wm_before=(%d,%d) wm_after=(%d,%d) allocs: %d->%d order_head=%s",
                self.tp_rank,
                alloc_id,
                getattr(decode_req.req, "bootstrap_room", "?"),
                _pre_wm[0],
                _pre_wm[1],
                post_wm[0],
                post_wm[1],
                _num_allocs_before,
                len(staging_allocator.allocations),
                (
                    staging_allocator.alloc_order[0]
                    if staging_allocator.alloc_order
                    else "empty"
                ),
            )
        receiver = decode_req.kv_receiver
        if (
            receiver is not None
            and hasattr(receiver, "bootstrap_infos")
            and receiver.bootstrap_infos
        ):
            wm_round, wm_tail = post_wm
            session_id = getattr(receiver, "session_id", "")
            for bootstrap_info in receiver.bootstrap_infos:
                try:
                    sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
                    with lock:
                        sock.send_multipart(
                            [
                                b"WATERMARK",
                                str(wm_round).encode("ascii"),
                                str(wm_tail).encode("ascii"),
                                session_id.encode("ascii"),
                            ]
                        )
                except Exception:
                    pass
            if _STAGING_DEBUG:
                logger.info(
                    "[WM-SEND] decode_tp=%d session=%s wm=(%d,%d) room=%s",
                    self.tp_rank,
                    session_id,
                    wm_round,
                    wm_tail,
                    getattr(decode_req.req, "bootstrap_room", "?"),
                )
        else:
            logger.warning(
                "[WM-SEND SKIP] decode_tp=%d alloc_id=%d room=%s receiver=%s "
                "has_bootstrap_infos=%s — watermark NOT sent to prefill",
                self.tp_rank,
                alloc_id,
                getattr(decode_req.req, "bootstrap_room", "?"),
                "None" if receiver is None else "present",
                (
                    hasattr(receiver, "bootstrap_infos")
                    and bool(getattr(receiver, "bootstrap_infos", None))
                    if receiver is not None
                    else False
                ),
            )

    def _complete_async_scatter(self, decode_req: DecodeRequest) -> None:
        """Wait for scatter event, then free staging and send watermark."""
        ctx = self.staging_ctx
        if ctx is not None:
            _, _, kv_buffer_info, _, _, _ = ctx
            device = kv_buffer_info["k_buffers"][0].device
            torch.cuda.default_stream(device).wait_event(decode_req._scatter_event)
        self._free_and_send_watermark(decode_req._scatter_alloc_id, decode_req)
        decode_req._scatter_event = None
        decode_req._scatter_alloc_id = -1

    def _process_pending_chunk_scatters(self):
        """Submit async scatter for CHUNK_READY tasks, tracked via per-req event lists."""
        ctx = self.staging_ctx
        if ctx is None:
            return
        kv_manager, staging_allocator, _, prefill_attn_tp, decode_attn_tp, _ = ctx
        pending = getattr(kv_manager, "pending_chunk_scatters", {})
        if not pending:
            return

        num_writers = (
            prefill_attn_tp // max(1, decode_attn_tp)
            if prefill_attn_tp > decode_attn_tp
            else 1
        )

        room_to_req = {
            dr.req.bootstrap_room: dr
            for dr in self.queue
            if dr.req.bootstrap_room in pending
        }

        for room, chunks in list(pending.items()):
            decode_req = room_to_req.get(room)
            if decode_req is None:
                continue
            chunk_infos = getattr(decode_req.kv_receiver, "chunk_staging_infos", [])
            if not chunk_infos:
                continue

            if not hasattr(decode_req, "_chunk_events"):
                decode_req._chunk_events = []

            by_chunk = defaultdict(list)
            for chunk in chunks:
                by_chunk[chunk[0]].append(chunk)

            scattered_chunks = set()
            for chunk_idx, group in by_chunk.items():
                if chunk_idx >= len(chunk_infos):
                    continue
                alloc_id, staging_offset, staging_round, _ = chunk_infos[chunk_idx]
                if staging_offset < 0:
                    continue
                if len(group) < num_writers:
                    continue

                page_start = group[0][1]
                num_pages = group[0][2]
                ok = self._scatter_staging_region(
                    staging_offset, page_start, num_pages, decode_req
                )
                if ok:
                    scatter_stream = getattr(staging_allocator, "_scatter_stream", None)
                    event = torch.cuda.Event()
                    if scatter_stream is not None:
                        event.record(scatter_stream)
                    decode_req._chunk_events.append((event, alloc_id))
                    chunk_infos[chunk_idx] = (-1, -1, 0, -1)
                    scattered_chunks.add(chunk_idx)

            if scattered_chunks:
                chunks[:] = [c for c in chunks if c[0] not in scattered_chunks]

        for decode_req in self.queue:
            chunk_events = getattr(decode_req, "_chunk_events", None)
            if not chunk_events:
                continue
            remaining = []
            for event, alloc_id in chunk_events:
                if event.query():
                    kv_buf = getattr(kv_manager, "kv_buffer_tensors", None)
                    if kv_buf is not None:
                        torch.cuda.default_stream(
                            kv_buf["k_buffers"][0].device
                        ).wait_event(event)
                    self._free_and_send_watermark(alloc_id, decode_req)
                else:
                    remaining.append((event, alloc_id))
            decode_req._chunk_events = remaining

    def _try_commit_and_finalize(
        self,
        decode_req: DecodeRequest,
        indices_to_remove,
        transferred_reqs,
        i: int,
    ):
        """Commit transfer and handle abort/success bookkeeping. Returns True if removed."""
        should_remove = self._commit_transfer_to_req(decode_req)
        if not should_remove:
            return False
        indices_to_remove.add(i)
        if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
            self.scheduler.stream_output(
                [decode_req.req], decode_req.req.return_logprob
            )
            release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
            if self.scheduler.enable_metrics:
                self.scheduler.metrics_collector.increment_transfer_failed_reqs()
        else:
            transferred_reqs.append(decode_req.req)
        return True

    def _advance_last_scatter(self, decode_req: DecodeRequest):
        """Advance LAST-SCATTER state machine for a single request.

        Called each poll when KVPoll.Success is detected.  Progresses through:
          1. Drain pending chunk scatters (while loop)
          2. Submit LAST-SCATTER to scatter_stream (async, non-blocking)
          3. Check event.query() → if done, free watermark and mark done

        Sets decode_req._staging_scatter_done = True when fully complete.
        No blocking synchronize — uses event.query() for non-blocking check.
        Commit is deferred to the poll_and_all_reduce-gated Phase 4.
        """
        if self.staging_ctx is None:
            decode_req._staging_scatter_done = True
            return

        # Step 1: drain pending chunk scatters for this room.
        # KVPoll.Success guarantees all prefill RDMA writes are done,
        # so CHUNK_READY messages will arrive shortly from decode_thread.
        kv_mgr = self.staging_ctx[0]
        room = decode_req.req.bootstrap_room
        pending = getattr(kv_mgr, "pending_chunk_scatters", {})

        if room in pending and pending[room]:
            self._process_pending_chunk_scatters()
            if room in pending and pending[room]:
                return

        # Step 2: submit LAST-SCATTER if not yet submitted.
        if not getattr(decode_req, "_staging_last_scatter_submitted", False):
            if _STAGING_DEBUG:
                logger.info(
                    "[LAST-SCATTER-TRIGGER] decode_tp=%d rid=%s room=%s "
                    "poll=Success session=%s",
                    self.tp_rank,
                    decode_req.req.rid,
                    room,
                    getattr(decode_req.kv_receiver, "session_id", ""),
                )
            slot_id = self._submit_scatter_staging(decode_req)
            if slot_id >= 0:
                scatter_stream = getattr(self.staging_ctx[1], "_scatter_stream", None)
                event = torch.cuda.Event()
                if scatter_stream is not None:
                    event.record(scatter_stream)
                decode_req._scatter_event = event
                decode_req._scatter_alloc_id = slot_id
                decode_req._staging_last_scatter_submitted = True
                if _STAGING_DEBUG:
                    logger.info(
                        "[LAST-SCATTER-SUBMITTED] decode_tp=%d rid=%s room=%s "
                        "alloc_id=%d",
                        self.tp_rank,
                        decode_req.req.rid,
                        room,
                        slot_id,
                    )
            else:
                if _STAGING_DEBUG:
                    logger.warning(
                        "[LAST-SCATTER-MISS] decode_tp=%d rid=%s room=%s "
                        "submit_returned=-1",
                        self.tp_rank,
                        decode_req.req.rid,
                        room,
                    )
                decode_req._staging_scatter_done = True
            return

        # Step 3: check if LAST-SCATTER event completed (non-blocking).
        event = getattr(decode_req, "_scatter_event", None)
        if event is not None and event.query():
            self._free_and_send_watermark(decode_req._scatter_alloc_id, decode_req)
            decode_req._scatter_event = None
            decode_req._scatter_alloc_id = -1
            decode_req._staging_scatter_done = True
            if _STAGING_DEBUG:
                logger.info(
                    "[LAST-SCATTER-DONE] decode_tp=%d rid=%s room=%s",
                    self.tp_rank,
                    decode_req.req.rid,
                    room,
                )

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        self._process_pending_chunk_scatters()

        # --- Phase 1: advance LAST-SCATTER state for requests that
        # reached KVPoll.Success.  Each rank progresses independently
        # (drain pending chunks → submit scatter → check event).
        # No blocking — uses event.query() for non-blocking completion
        # check.  The actual commit is gated by poll_and_all_reduce in
        # Phase 4, ensuring all TP ranks commit together.
        if self.staging_ctx is not None:
            for decode_req in self.queue:
                if not getattr(decode_req, "_staging_scatter_done", False):
                    raw_poll = decode_req.kv_receiver.poll()
                    if raw_poll == KVPoll.Success:
                        self._advance_last_scatter(decode_req)

        # --- Phase 2: guard check — ensure all TP ranks have same queue size
        n = len(self.queue)
        guard = torch.tensor(
            [1 if self.queue else 0, n, -n],
            dtype=torch.int64,
            device="cpu",
        )
        dist.all_reduce(guard, op=dist.ReduceOp.MIN, group=self.gloo_group)
        if guard[0].item() == 0 or guard[1].item() != -guard[2].item():
            return []

        # --- Phase 3: poll_and_all_reduce with adjusted polls.
        # For staging requests, demote Success → Transferring until the
        # local scatter is done.  all_reduce(MIN) ensures commit only
        # happens when ALL TP ranks have completed their scatter.
        raw_polls = [int(dr.kv_receiver.poll()) for dr in self.queue]
        if self.staging_ctx is not None:
            for i, decode_req in enumerate(self.queue):
                if raw_polls[i] == int(KVPoll.Success):
                    if not getattr(decode_req, "_staging_scatter_done", False):
                        raw_polls[i] = int(KVPoll.Transferring)
        poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
        dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=self.gloo_group)
        polls = poll_tensor.tolist()

        # --- Phase 4: process poll results.  commit only when
        # all_reduce confirms all TP ranks are ready.
        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Failed:
                error_message = f"Decode transfer failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.stream_output(
                    [decode_req.req], decode_req.req.return_logprob
                )
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                indices_to_remove.add(i)
                if self.scheduler.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                self._try_commit_and_finalize(
                    decode_req,
                    indices_to_remove,
                    transferred_reqs,
                    i,
                )
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs


class SchedulerDisaggregationDecodeMixin:

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # polling and allocating kv cache
            self.process_decode_queue()

            # Get the next batch to run
            batch = self.get_next_disagg_decode_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.self_check_during_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    def get_next_disagg_decode_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        """Create fake completed prefill if possible and merge with running batch"""
        # Merge the prefill batch into the running batch
        last_batch = self.last_batch
        if last_batch and last_batch.forward_mode.is_prebuilt():
            # chunked prefill doesn't happen in decode instance.
            assert self.chunked_req is None
            # Filter finished batches.
            last_batch.filter_batch()
            if not last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = last_batch
                else:
                    # merge running_batch with prefill batch
                    self.running_batch.merge_batch(last_batch)

        new_prebuilt_batch = self.get_new_prebuilt_batch()

        ret: Optional[ScheduleBatch] = None
        if new_prebuilt_batch:
            ret = new_prebuilt_batch
        else:
            if self.running_batch.is_empty():
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None

        # 1. decode + None -> decode + idle
        # 2. decode + prebuilt -> decode + idle (idle forward, prebuilt returns)
        # 3. prebuilt + None -> None (None forward, prebuilt returns) + None
        # 4. prebuilt + decode + None -> idle (idle forward, prebuilt returns) + decode + idle
        ret = self.maybe_prepare_mlp_sync_batch(ret)

        if ret:
            set_schedule_time_batch(ret)
        return ret

    def get_new_prebuilt_batch(self: Scheduler) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        curr_batch_size = self.running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                req.init_next_round_input(self.tree_cache)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.server_args.disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = (
                self.server_args.disaggregation_decode_polling_interval
            )

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            self.waiting_queue.extend(transferred_reqs)
