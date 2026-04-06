# Async TP Without `_symmetric_memory`

This note sketches a replacement async tensor-parallel prototype that does not
rely on PyTorch's `_symmetric_memory` backend.

## Goal

Preserve the useful part of the async TP idea:

- overlap TP communication with GEMM
- keep tensors in a scattered layout between compatible layers

Avoid the current backend costs:

- `symm_mem_barrier` synchronization overhead
- implicit chunking inside fused private kernels
- opaque internal `copy_` / layout transformations

## Design Principles

1. Use explicit streams.
   - one compute stream
   - one communication stream

2. Use explicit buffers.
   - double-buffer or triple-buffer per layer stage
   - no hidden symmetric-memory workspace protocol

3. Keep chunking under SGLang control.
   - choose chunk size in Python / runtime
   - do not let the backend decide an opaque micro-pipeline shape

4. Preserve large GEMMs where possible.
   - do not over-fragment GEMMs purely to force overlap

## Candidate Execution Model

### Column-parallel path

Target operation:

- gather scattered activations
- run local GEMM

Prototype:

1. Split activations into token chunks
2. Launch `all_gather` for chunk `i + 1` on the comm stream
3. Run GEMM for chunk `i` on the compute stream
4. Use events to hand gathered chunk buffers from comm to compute

### Row-parallel path

Target operation:

- run local GEMM
- reduce-scatter partial output

Prototype:

1. Run local GEMM for chunk `i` on the compute stream
2. Record an event for chunk `i`
3. Launch `reduce_scatter` for chunk `i` on the comm stream once the event is ready
4. Optionally begin GEMM for chunk `i + 1` while the comm stream handles chunk `i`

## Minimal Runtime Abstraction

```text
AsyncTpChunk {
  chunk_id
  token_start
  token_end
  input_buffer
  gathered_buffer
  gemm_output_buffer
  scattered_output_buffer
  comm_ready_event
  compute_ready_event
}
```

## Suggested First Prototype

Start with the row-parallel MLP down-projection path only.

Why:

- narrow surface area
- easier to isolate than attention
- direct comparison against baseline all-reduce / reduce-scatter

Prototype steps:

1. Select a fixed chunk size on the token dimension
2. Compute local GEMM chunk on the compute stream
3. Launch NCCL `reduce_scatter` for the completed chunk on the comm stream
4. Feed the scattered output chunk to the next compatible layer

## Early Success Criteria

The prototype is worth continuing only if it shows all of the following:

1. No correctness regressions across divisible and non-divisible token counts
2. No large new barrier-style hotspot
3. Kernel count stays close to baseline
4. GEMM efficiency does not collapse due to over-fragmentation

## Instrumentation To Add

To make the prototype debuggable, log or trace:

- chunk count per layer
- chunk size
- comm stream wait duration
- compute stream idle duration
- NCCL collective duration per chunk
- time from chunk GEMM completion to chunk communication completion

## Open Questions

1. What chunk size preserves GEMM efficiency on H100 while still allowing useful overlap?
2. Should attention and MLP share the same chunk size?
3. Is it better to keep the scattered layout across multiple layers or only within a single block?
4. Can we reuse the MoE chunk-pipeline scaffolding as a generic async TP scheduler skeleton?
