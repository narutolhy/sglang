"""
FX pass to replace AllReduce with ReduceScatter + AllGather for
sequence parallelism in the compile path.

In PCG mode, AllReduce appears as `outplace_all_reduce` (custom_op,
not a split_op). This pass replaces it with RS+AG, allowing
intermediate ops (LayerNorm, activation, residual) to run on
scattered [M/tp, H] tensors — saving 4x memory bandwidth.

The pass operates on the FX graph BEFORE split_graph() in
SGLangBackend.__call__.

Pattern matched (per transformer layer):
  GEMM → outplace_all_reduce → fused_add_rmsnorm → GEMM

Replaced with:
  GEMM → reduce_scatter → fused_add_rmsnorm([M/tp,H]) → all_gather → GEMM

Benefits:
  - LayerNorm/activation run on [M/tp, H] = 4x less bandwidth
  - RS+AG communication volume ≈ AllReduce (no regression)
  - Future: fused GEMM+RS / AG+GEMM for overlap
"""

import logging
import os

import torch
from torch import fx

logger = logging.getLogger(__name__)


def is_async_tp_pass_enabled() -> bool:
    """Check if the async TP FX pass should be applied."""
    return os.environ.get("SGLANG_ENABLE_ASYNC_TP_FX_PASS", "0") == "1"


def async_tp_pass(graph: fx.GraphModule) -> int:
    """Replace outplace_all_reduce with reduce_scatter + all_gather.

    This replaces allreduce with RS+AG, keeping the same semantics
    (output tensor has the fully-reduced values). The benefit comes
    when Phase 2 moves ops between RS and AG to operate on scattered
    tensors.

    For now (Phase 1), this is a simple 1:1 replacement to verify
    correctness and measure the overhead of RS+AG vs AllReduce.

    Returns: number of AllReduce nodes replaced.
    """
    from sglang.srt.distributed import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return 0

    replacements = 0

    for node in list(graph.graph.nodes):
        if node.op != "call_function":
            continue

        target_name = str(node.target)

        # Match both inplace and outplace all_reduce
        is_inplace = "inplace_all_reduce" in target_name
        is_outplace = "outplace_all_reduce" in target_name and "inplace" not in target_name

        if not (is_inplace or is_outplace):
            continue

        if is_inplace:
            # inplace_all_reduce(tensor, group_name) -> None
            input_tensor_node = node.args[0]
            group_name_node = node.args[1] if len(node.args) > 1 else node.kwargs.get("group_name")
        else:
            # outplace_all_reduce(tensor, group_name, method) -> tensor
            input_tensor_node = node.args[0]
            group_name_node = node.args[1] if len(node.args) > 1 else node.kwargs.get("group_name")

        if group_name_node is None:
            logger.warning(f"Cannot find group_name for {target_name}, skipping")
            continue

        with graph.graph.inserting_before(node):
            # 1. Create RS output buffer [M/tp, H]
            rs_output = graph.graph.call_function(
                torch.ops.sglang.async_tp_create_rs_buffer,
                args=(input_tensor_node, tp_size),
            )

            # 2. reduce_scatter(rs_output, input, group_name)
            graph.graph.call_function(
                torch.ops.sglang.reg_reduce_scatter_tensor,
                args=(rs_output, input_tensor_node, group_name_node),
            )

            if is_outplace:
                # 3. Create AG output buffer [M, H] (outplace returns new tensor)
                ag_output = graph.graph.call_function(
                    torch.ops.sglang.async_tp_create_ag_buffer,
                    args=(rs_output, tp_size),
                )

                # 4. all_gather(ag_output, rs_output, group_name)
                graph.graph.call_function(
                    torch.ops.sglang.reg_all_gather_into_tensor,
                    args=(ag_output, rs_output, group_name_node),
                )

                # Replace all uses of outplace_all_reduce result with ag_output
                node.replace_all_uses_with(ag_output)
            else:
                # For inplace: all_gather back into original tensor
                graph.graph.call_function(
                    torch.ops.sglang.reg_all_gather_into_tensor,
                    args=(input_tensor_node, rs_output, group_name_node),
                )

        graph.graph.erase_node(node)
        replacements += 1

    if replacements > 0:
        graph.graph.lint()
        graph.recompile()
        logger.info(
            f"AsyncTP FX pass: replaced {replacements} all_reduce ops "
            f"with reduce_scatter + all_gather (tp_size={tp_size})"
        )

    return replacements


def register_async_tp_ops():
    """Register helper ops needed by the FX pass."""
    from sglang.srt.utils.custom_op import register_custom_op

    @register_custom_op(
        op_name="async_tp_create_rs_buffer",
        mutates_args=[],
        fake_impl=_fake_create_rs_buffer,
    )
    def _create_rs_buffer_impl(
        tensor: torch.Tensor, tp_size: int
    ) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[0] = shape[0] // tp_size
        return tensor.new_empty(shape)

    @register_custom_op(
        op_name="async_tp_create_ag_buffer",
        mutates_args=[],
        fake_impl=_fake_create_ag_buffer,
    )
    def _create_ag_buffer_impl(
        scattered: torch.Tensor, tp_size: int
    ) -> torch.Tensor:
        shape = list(scattered.shape)
        shape[0] = shape[0] * tp_size
        return scattered.new_empty(shape)


def _fake_create_rs_buffer(tensor: torch.Tensor, tp_size: int) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[0] = shape[0] // tp_size
    return tensor.new_empty(shape)


def _fake_create_ag_buffer(scattered: torch.Tensor, tp_size: int) -> torch.Tensor:
    shape = list(scattered.shape)
    shape[0] = shape[0] * tp_size
    return scattered.new_empty(shape)


_ops_registered = False


def ensure_ops_registered():
    global _ops_registered
    if _ops_registered:
        return
    register_async_tp_ops()
    _ops_registered = True
