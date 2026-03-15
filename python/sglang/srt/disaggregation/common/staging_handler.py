"""
Staging handler for heterogeneous TP KV cache transfer.

Isolates staging scatter lifecycle from decode.py and conn.py.
Generic (backend-agnostic) code is at the top; mooncake-specific
protocol code is at the bottom.
"""

from __future__ import annotations

import dataclasses
import logging
import struct
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest


# ======================================================================
# Generic staging state and handler (backend-agnostic)
# ======================================================================


@dataclasses.dataclass
class DecodeStagingState:
    """Staging-specific state for decode mode."""

    allocator: object = None
    pending_chunk_scatters: dict = dataclasses.field(default_factory=dict)
    room_bootstrap: dict = dataclasses.field(default_factory=dict)
    room_receivers: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PrefillStagingState:
    """Staging-specific state for prefill mode."""

    buffers: list = dataclasses.field(default_factory=list)
    remote_watermarks: dict = dataclasses.field(default_factory=dict)
    watermark_cv: threading.Condition = dataclasses.field(
        default_factory=threading.Condition
    )
    prefetch_requested: set = dataclasses.field(default_factory=set)
    prefetch_sockets: dict = dataclasses.field(default_factory=dict)


class DecodeStagingHandler:
    """Decode-side staging scatter lifecycle manager."""

    def __init__(
        self,
        kv_manager,
        staging_allocator,
        kv_buffer_info: dict,
        prefill_tp: int,
        decode_tp: int,
        total_kv_heads: int,
        tp_rank: int,
        scheduler,
    ):
        self.kv_manager = kv_manager
        self.staging_allocator = staging_allocator
        self.kv_buffer_info = kv_buffer_info
        self.prefill_tp = prefill_tp
        self.decode_tp = decode_tp
        self.total_kv_heads = total_kv_heads
        self.tp_rank = tp_rank
        self.scheduler = scheduler

    @classmethod
    def try_create(cls, scheduler, tp_rank: int) -> Optional["DecodeStagingHandler"]:
        """Factory: create handler if staging is enabled and heterogeneous TP detected."""
        kv_manager = getattr(scheduler, "_decode_kv_manager", None)
        if kv_manager is None:
            return None
        _stg = getattr(kv_manager, "_staging", None)
        staging_allocator = getattr(_stg, "allocator", None) if _stg else None
        if staging_allocator is None:
            return None
        kv_buffer_info = getattr(kv_manager, "kv_buffer_tensors", None)
        if kv_buffer_info is None:
            return None
        prefill_tp = getattr(kv_manager, "prefill_attn_tp_size", 0)
        decode_tp = kv_manager.attn_tp_size
        if prefill_tp == 0 or prefill_tp == decode_tp:
            return None

        from sglang.srt.disaggregation.common.staging import resolve_total_kv_heads

        total_kv_heads = resolve_total_kv_heads(kv_manager.kv_args, decode_tp)
        return cls(
            kv_manager=kv_manager,
            staging_allocator=staging_allocator,
            kv_buffer_info=kv_buffer_info,
            prefill_tp=prefill_tp,
            decode_tp=decode_tp,
            total_kv_heads=total_kv_heads,
            tp_rank=tp_rank,
            scheduler=scheduler,
        )

    # ------------------------------------------------------------------
    # Public interface used by DecodeTransferQueue.pop_transferred
    # ------------------------------------------------------------------

    def is_done(self, decode_req: "DecodeRequest") -> bool:
        """Return True if staging scatter is complete for this request."""
        return getattr(decode_req, "_staging_scatter_done", False)

    def advance_scatter(self, decode_req: "DecodeRequest", queue: list) -> None:
        """Advance scatter state machine for one request after KVPoll.Success.

        Progresses through:
          1. Drain pending chunk scatters
          2. Submit last-chunk scatter to scatter_stream (async)
          3. Check event completion -> free staging allocation and send watermark
        """
        room = decode_req.req.bootstrap_room
        pending = self.kv_manager._staging.pending_chunk_scatters

        if room in pending and pending[room]:
            self._process_pending_chunk_scatters(queue)
            if room in pending and pending[room]:
                return

        if not getattr(decode_req, "_staging_last_scatter_submitted", False):
            slot_id = self._submit_last_scatter(decode_req)
            if slot_id >= 0:
                event = torch.cuda.Event()
                event.record(self.staging_allocator._scatter_stream)
                decode_req._scatter_event = event
                decode_req._scatter_alloc_id = slot_id
                decode_req._staging_last_scatter_submitted = True
            else:
                decode_req._staging_scatter_done = True
            return

        event = getattr(decode_req, "_scatter_event", None)
        if event is not None and event.query():
            self._free_and_send_watermark(decode_req._scatter_alloc_id, decode_req)
            decode_req._scatter_event = None
            decode_req._scatter_alloc_id = -1
            decode_req._staging_scatter_done = True

    def process_pending_chunks(self, queue: list) -> None:
        """Process CHUNK_READY messages and submit async scatter."""
        self._process_pending_chunk_scatters(queue)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: "DecodeRequest",
    ) -> bool:
        """Submit scatter kernels for a staging region to scatter_stream."""
        from sglang.srt.disaggregation.common.staging import scatter_staging_to_kv

        k_buffers = self.kv_buffer_info["k_buffers"]
        v_buffers = self.kv_buffer_info["v_buffers"]
        page_size = self.kv_buffer_info["page_size"]
        dst_tp_rank = self.kv_manager.kv_args.engine_rank % self.decode_tp

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

        staging_view = self.staging_allocator.buffer.buffer[staging_offset:]

        if not hasattr(self.staging_allocator, "_scatter_stream"):
            self.staging_allocator._scatter_stream = torch.cuda.Stream()

        staging_view[0].item()

        with torch.cuda.stream(self.staging_allocator._scatter_stream):
            scatter_staging_to_kv(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                self.prefill_tp,
                self.decode_tp,
                dst_tp_rank,
                self.total_kv_heads,
            )

        return True

    def _submit_last_scatter(self, decode_req: "DecodeRequest") -> int:
        """Submit scatter for the last chunk. Returns alloc_id >= 0, or -1."""
        receiver = decode_req.kv_receiver
        chunk_infos = getattr(receiver, "chunk_staging_infos", [])
        if not chunk_infos:
            return -1

        last_info = chunk_infos[-1]
        alloc_id, staging_offset, _, _ = last_info
        if staging_offset < 0 or alloc_id < 0:
            return -1

        seq_len = len(decode_req.req.origin_input_ids)
        ps = self.scheduler.token_to_kv_pool_allocator.page_size
        total_pages = (seq_len + ps - 1) // ps

        n = len(chunk_infos)
        prefill_cps = (
            getattr(self.kv_manager, "prefill_chunked_prefill_size", None)
            or self.scheduler.server_args.chunked_prefill_size
            or 8192
        )
        chunk_pages = max(1, prefill_cps // ps)
        page_start = chunk_pages * (n - 1)
        last_num_pages = total_pages - page_start

        ok = self._scatter_region(
            staging_offset, page_start, last_num_pages, decode_req
        )
        return alloc_id if ok else -1

    def _free_and_send_watermark(
        self, alloc_id: int, decode_req: "DecodeRequest"
    ) -> None:
        """Free a staging allocation and send watermark to prefill."""
        self.staging_allocator.free(alloc_id)
        post_wm = self.staging_allocator.get_watermark()
        receiver = decode_req.kv_receiver
        if receiver is not None and receiver.bootstrap_infos:
            wm_round, wm_tail = post_wm
            session_id = receiver.session_id
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

    def _process_pending_chunk_scatters(self, queue: list) -> None:
        """Submit async scatter for CHUNK_READY tasks."""
        pending = self.kv_manager._staging.pending_chunk_scatters
        if not pending:
            return

        num_writers = (
            self.prefill_tp // max(1, self.decode_tp)
            if self.prefill_tp > self.decode_tp
            else 1
        )

        room_to_req = {
            dr.req.bootstrap_room: dr
            for dr in queue
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
                ok = self._scatter_region(
                    staging_offset, page_start, num_pages, decode_req
                )
                if ok:
                    event = torch.cuda.Event()
                    event.record(self.staging_allocator._scatter_stream)
                    decode_req._chunk_events.append((event, alloc_id))
                    chunk_infos[chunk_idx] = (-1, -1, 0, -1)
                    scattered_chunks.add(chunk_idx)

            if scattered_chunks:
                chunks[:] = [c for c in chunks if c[0] not in scattered_chunks]

        for decode_req in queue:
            chunk_events = getattr(decode_req, "_chunk_events", None)
            if not chunk_events:
                continue
            remaining = []
            for event, alloc_id in chunk_events:
                if event.query():
                    torch.cuda.default_stream(
                        self.kv_buffer_info["k_buffers"][0].device
                    ).wait_event(event)
                    self._free_and_send_watermark(alloc_id, decode_req)
                else:
                    remaining.append((event, alloc_id))
            decode_req._chunk_events = remaining


def is_watermark_ready(
    staging_state, session_id: str, alloc_round: int, alloc_end: int
) -> bool:
    """Non-blocking check: is the staging region safe to write?"""
    if alloc_round <= 0:
        return True
    prev_round = alloc_round - 1
    wm_round, wm_tail = staging_state.remote_watermarks.get(session_id, (0, 0))
    return prev_round < wm_round or (prev_round == wm_round and alloc_end <= wm_tail)


def allocate_chunk_staging_for_receiver(kv_mgr, kv_indices) -> list:
    """Allocate per-chunk staging regions from the ring buffer allocator."""
    _stg = getattr(kv_mgr, "_staging", None)
    staging_allocator = getattr(_stg, "allocator", None) if _stg else None
    if staging_allocator is None:
        return []

    from sglang.srt.disaggregation.common.staging import (
        allocate_chunk_staging,
        resolve_total_kv_heads,
    )

    page_size = kv_mgr.kv_args.page_size
    kv_item_lens = kv_mgr.kv_args.kv_item_lens
    num_kv_layers = len(kv_item_lens) // 2
    decode_bytes_per_token = kv_item_lens[0] // page_size
    attn_tp_size = kv_mgr.attn_tp_size
    prefill_attn_tp = getattr(kv_mgr, "prefill_attn_tp_size", attn_tp_size)
    total_kv_heads = resolve_total_kv_heads(kv_mgr.kv_args, attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // max(1, attn_tp_size))
    bytes_per_head_per_token = decode_bytes_per_token // dst_heads_per_rank
    dst_tp_rank = kv_mgr.kv_args.engine_rank % max(1, attn_tp_size)

    chunked_prefill_size = (
        getattr(kv_mgr, "prefill_chunked_prefill_size", None)
        or kv_mgr.server_args.chunked_prefill_size
        or 8192
    )
    chunk_pages = max(1, chunked_prefill_size // page_size)

    return allocate_chunk_staging(
        staging_allocator,
        len(kv_indices),
        page_size,
        chunk_pages,
        prefill_attn_tp,
        attn_tp_size,
        dst_tp_rank,
        total_kv_heads,
        bytes_per_head_per_token,
        num_kv_layers,
    )


# ======================================================================
# Mooncake-specific staging protocol and utilities
# ======================================================================


@dataclasses.dataclass
class StagingTransferInfo:
    """Per-chunk staging allocation info attached to a TransferInfo."""

    offsets: List[int] = dataclasses.field(default_factory=lambda: [-1])
    rounds: List[int] = dataclasses.field(default_factory=lambda: [0])
    ends: List[int] = dataclasses.field(default_factory=lambda: [-1])

    @staticmethod
    def _parse_csv_ints(raw: str, default: int) -> List[int]:
        if not raw or raw == "":
            return [default]
        return [int(x) for x in raw.split(",")]

    def set_chunk(self, idx: int, offset: int, rnd: int, end: int):
        while len(self.offsets) <= idx:
            self.offsets.append(-1)
            self.rounds.append(0)
            self.ends.append(-1)
        self.offsets[idx] = offset
        self.rounds[idx] = rnd
        self.ends[idx] = end

    def to_zmq_fields(self) -> Tuple[bytes, bytes, bytes]:
        return (
            ",".join(str(x) for x in self.offsets).encode("ascii"),
            ",".join(str(x) for x in self.rounds).encode("ascii"),
            ",".join(str(x) for x in self.ends).encode("ascii"),
        )

    @classmethod
    def from_zmq_fields(cls, msg: list) -> Optional["StagingTransferInfo"]:
        offsets_raw = msg[8].decode("ascii") if len(msg) > 8 and msg[8] != b"" else ""
        rounds_raw = msg[9].decode("ascii") if len(msg) > 9 and msg[9] != b"" else ""
        ends_raw = msg[10].decode("ascii") if len(msg) > 10 and msg[10] != b"" else ""
        if not offsets_raw and not rounds_raw and not ends_raw:
            return None
        return cls(
            offsets=cls._parse_csv_ints(offsets_raw, -1),
            rounds=cls._parse_csv_ints(rounds_raw, 0),
            ends=cls._parse_csv_ints(ends_raw, -1),
        )


@dataclasses.dataclass
class StagingRegisterInfo:
    """Staging buffer registration info attached to a KVArgsRegisterInfo."""

    base_ptr: int = 0
    total_size: int = 0

    @classmethod
    def from_zmq_fields(cls, msg: list) -> Optional["StagingRegisterInfo"]:
        base_ptr = (
            struct.unpack("Q", msg[12])[0] if len(msg) > 12 and len(msg[12]) == 8 else 0
        )
        total_size = (
            int(msg[13].decode("ascii")) if len(msg) > 13 and len(msg[13]) > 0 else 0
        )
        if base_ptr == 0 and total_size == 0:
            return None
        return cls(base_ptr=base_ptr, total_size=total_size)


class PrefillStagingStrategy:
    """Prefill-side staging transfer: readiness check + gather-RDMA execution.

    Encapsulates the decision logic (chunk index calculation, staging offset
    lookup, watermark readiness) and delegates actual RDMA to the kv_manager.
    """

    def __init__(self, kv_manager, staging_buffer):
        self.kv_manager = kv_manager
        self.staging_buffer = staging_buffer
        page_size = kv_manager.kv_buffer_tensors["page_size"]
        cps = kv_manager.server_args.chunked_prefill_size or 8192
        self.full_chunk_pages = max(1, cps // page_size)

    def check_ready(
        self,
        req,
        kv_chunk_index_start: int,
        num_chunk_pages: int,
    ) -> Tuple[bool, int, int, int, int]:
        """Check if staging offset and watermark are ready for this chunk.

        Returns (ready, chunk_idx, offset, round, end).
        offset == ALLOC_OVERSIZED means permanent failure (fall back to slice).
        offset == -1 means allocation pending (re-enqueue).
        """
        from sglang.srt.disaggregation.common.staging import StagingAllocator

        chunk_idx = (
            kv_chunk_index_start // self.full_chunk_pages
            if self.full_chunk_pages > 0
            else 0
        )

        stg = req.staging
        if stg is None or chunk_idx >= len(stg.offsets):
            return (False, chunk_idx, -1, 0, -1)

        c_offset = stg.offsets[chunk_idx]
        if c_offset == StagingAllocator.ALLOC_OVERSIZED:
            return (False, chunk_idx, StagingAllocator.ALLOC_OVERSIZED, 0, -1)
        if c_offset < 0:
            return (False, chunk_idx, -1, 0, -1)

        c_round = stg.rounds[chunk_idx]
        c_end = stg.ends[chunk_idx]

        if not self.kv_manager._is_watermark_ready(
            req.mooncake_session_id, c_round, c_end
        ):
            return (False, chunk_idx, c_offset, c_round, c_end)

        return (True, chunk_idx, c_offset, c_round, c_end)

    def transfer(
        self,
        session_id: str,
        prefill_kv_indices,
        dst_staging_ptr: int,
        dst_staging_size: int,
        target_info,
    ) -> int:
        """Execute staged transfer (gather + RDMA).

        Returns 0 on success, -1 to signal fallback to slice path.
        """
        try:
            return self.kv_manager.send_kvcache_staged(
                session_id,
                prefill_kv_indices,
                dst_staging_ptr,
                dst_staging_size,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                staging_buffer=self.staging_buffer,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Staging] KV transfer via staging buffer failed: {e}. "
                f"session={session_id}"
            ) from e


def init_staging_buffers(engine, kv_args, count: int) -> list:
    """Create prefill-side staging buffers and register them with the engine.

    Returns list of StagingBuffer instances.
    """
    from sglang.srt.disaggregation.common.staging import StagingBuffer
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )
    from sglang.srt.environ import envs

    size_mb = envs.SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB.get()
    size_bytes = size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    _, custom_mem_pool, pool_type = init_mooncake_custom_mem_pool(device)
    if custom_mem_pool is None:
        logger.warning(
            "No mooncake custom mem pool available for staging buffer. "
            "NVLink transport will NOT work. Set SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        engine.batch_register([buf.get_ptr()], [buf.get_size()])
        buffers.append(buf)
    return buffers


def init_staging_allocator(engine, kv_args):
    """Create decode-side staging ring-buffer allocator and register with engine.

    Returns a StagingAllocator instance.
    """
    from sglang.srt.disaggregation.common.staging import StagingAllocator
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )
    from sglang.srt.environ import envs

    pool_size_mb = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get()
    pool_size_bytes = pool_size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    _, custom_mem_pool, _ = init_mooncake_custom_mem_pool(device)
    allocator = StagingAllocator(pool_size_bytes, device, gpu_id, custom_mem_pool)
    engine.batch_register([allocator.get_base_ptr()], [allocator.get_total_size()])
    return allocator


def handle_staging_req(
    msg,
    staging_allocator,
    kv_args,
    attn_tp_size: int,
    prefill_attn_tp_size: int,
    kv_buffer_tensors,
    room_receivers: dict,
    room_bootstrap: dict,
):
    """Allocate staging for a chunk on-demand and send STAGING_RSP to prefill.

    Deduplicates: multiple prefill TP ranks requesting the same (room, chunk_idx)
    only allocate once.  Sends ALLOC_OVERSIZED on permanent failure.
    """
    from sglang.srt.disaggregation.common.staging import StagingAllocator

    room = int(msg[1].decode("ascii"))
    chunk_idx = int(msg[2].decode("ascii"))
    chunk_num_pages = int(msg[3].decode("ascii"))
    session_id = msg[4].decode("ascii")

    if staging_allocator is None:
        return

    receiver = room_receivers.get(room)
    if receiver is None:
        return
    infos = getattr(receiver, "chunk_staging_infos", [])

    if chunk_idx < len(infos) and infos[chunk_idx][0] >= 0:
        _, offset, rnd, end = infos[chunk_idx]
    elif (
        chunk_idx < len(infos)
        and infos[chunk_idx][1] == StagingAllocator.ALLOC_OVERSIZED
    ):
        offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
    else:
        from sglang.srt.disaggregation.common.staging import (
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        page_size = kv_args.page_size
        kv_item_lens = kv_args.kv_item_lens
        num_kv_layers = len(kv_item_lens) // 2
        decode_bytes_per_token = kv_item_lens[0] // page_size
        total_kv_heads = resolve_total_kv_heads(kv_args, attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // max(1, attn_tp_size))
        bytes_per_head_per_token = decode_bytes_per_token // dst_heads_per_rank
        dst_tp_rank = kv_args.engine_rank % max(1, attn_tp_size)

        chunk_tokens = chunk_num_pages * page_size
        _, _, required = compute_staging_layout(
            prefill_attn_tp_size,
            attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            chunk_tokens,
            bytes_per_head_per_token,
            num_kv_layers,
        )
        result = staging_allocator.assign(required)
        if result is None:
            logger.error(
                "[STAGING_REQ] alloc failed room=%s chunk=%d (need %d bytes, "
                "buffer total=%d bytes). Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB.",
                room,
                chunk_idx,
                required,
                staging_allocator.total_size,
            )
            offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1))
            infos[chunk_idx] = (-1, StagingAllocator.ALLOC_OVERSIZED, 0, -1)
        else:
            alloc_id, offset, rnd = result
            end = offset + required
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1))
            infos[chunk_idx] = (alloc_id, offset, rnd, end)

    bootstrap_infos = room_bootstrap.get(room)
    if bootstrap_infos:
        for bi in bootstrap_infos:
            try:
                sock, lock = receiver._connect_to_bootstrap_server(bi)
                with lock:
                    sock.send_multipart(
                        [
                            b"STAGING_RSP",
                            str(room).encode("ascii"),
                            str(chunk_idx).encode("ascii"),
                            str(offset).encode("ascii"),
                            str(rnd).encode("ascii"),
                            str(end).encode("ascii"),
                            session_id.encode("ascii"),
                        ]
                    )
            except Exception:
                pass


def prefetch_staging_reqs(
    room: int,
    transfer_infos: dict,
    kv_buffer_tensors: dict,
    chunked_prefill_size: int,
    staging_requested: set,
    prefetch_sockets: dict,
) -> None:
    """Send STAGING_REQ for all chunks before the prefill forward starts.

    Called from the scheduler right after batch formation, so that decode
    allocates staging during the GPU forward pass.
    """
    import zmq

    from sglang.srt.utils import format_tcp_address, is_valid_ipv6_address

    page_size = kv_buffer_tensors["page_size"]
    cps = chunked_prefill_size or 8192
    full_chunk_pages = max(1, cps // page_size)

    for session_id, tinfo in transfer_infos[room].items():
        if tinfo.is_dummy:
            continue
        total_pages = len(tinfo.dst_kv_indices)
        if total_pages == 0:
            continue
        num_chunks = (total_pages + full_chunk_pages - 1) // full_chunk_pages

        for chunk_idx in range(num_chunks):
            stg_key = (room, chunk_idx, session_id)
            if stg_key in staging_requested:
                continue
            staging_requested.add(stg_key)

            remaining = total_pages - chunk_idx * full_chunk_pages
            chunk_pages = min(full_chunk_pages, remaining)
            try:
                ep = format_tcp_address(tinfo.endpoint, tinfo.dst_port)
                if ep not in prefetch_sockets:
                    sock = zmq.Context().socket(zmq.PUSH)
                    if is_valid_ipv6_address(tinfo.endpoint):
                        sock.setsockopt(zmq.IPV6, 1)
                    sock.connect(ep)
                    prefetch_sockets[ep] = sock
                prefetch_sockets[ep].send_multipart(
                    [
                        b"STAGING_REQ",
                        str(room).encode("ascii"),
                        str(chunk_idx).encode("ascii"),
                        str(chunk_pages).encode("ascii"),
                        session_id.encode("ascii"),
                    ]
                )
            except Exception:
                staging_requested.discard(stg_key)
