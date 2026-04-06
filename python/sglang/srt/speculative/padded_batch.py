"""Padded batch utilities for speculative decoding.

Pads variable-length inputs to uniform shape to enable:
1. FlashMLA / high-performance attention kernels (require uniform query length)
2. Full CUDA graph capture (require fixed tensor shapes)

Example:
  Before padding (varlen):
    Req 0: [S0] [S1]            (2 tokens - short prefill)
    Req 1: [S0] [S1] [S2] [S3] (4 tokens - full decode)
    Req 2: [S0] [S1] [S2]      (3 tokens - partial accept)
    Req 3: [S0]                 (1 token - min accept)

  After padding (uniform, max_len=4):
    Req 0: [S0] [S1] [X]  [X]   (X = pad token)
    Req 1: [S0] [S1] [S2] [S3]
    Req 2: [S0] [S1] [S2] [X]
    Req 3: [S0] [X]  [X]  [X]
"""

import torch
from typing import Optional, Tuple

# Pad token ID: use 0 (will be ignored by attention mask via seq_lens)
PAD_TOKEN_ID = 0
# Padding slot ID for KV cache: -1 means skip write
PADDING_SLOT_ID = -1


def pad_batch_to_uniform(
    input_ids: torch.Tensor,        # [total_tokens] flat
    positions: torch.Tensor,        # [total_tokens] flat
    out_cache_loc: torch.Tensor,    # [total_tokens] flat
    extend_seq_lens: torch.Tensor,  # [bs] per-request token counts
    extend_prefix_lens: torch.Tensor,  # [bs] prefix lengths
    max_len: int,                   # target uniform length (e.g., num_spec_tokens + 1)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad variable-length batch to uniform shape.

    Returns:
        padded_input_ids: [bs * max_len]
        padded_positions: [bs * max_len]
        padded_out_cache_loc: [bs * max_len] (padding slots = PADDING_SLOT_ID)
        pad_mask: [bs * max_len] bool (True = valid, False = padding)
        total_padded_tokens: bs * max_len
    """
    device = input_ids.device
    bs = extend_seq_lens.shape[0]
    total_padded = bs * max_len

    # Allocate padded tensors
    padded_input_ids = torch.full((total_padded,), PAD_TOKEN_ID, dtype=input_ids.dtype, device=device)
    padded_positions = torch.zeros(total_padded, dtype=positions.dtype, device=device)
    padded_out_cache_loc = torch.full((total_padded,), PADDING_SLOT_ID, dtype=out_cache_loc.dtype, device=device)
    pad_mask = torch.zeros(total_padded, dtype=torch.bool, device=device)

    # Fill valid tokens
    # extend_start_loc = cumsum([0] + extend_seq_lens[:-1])
    start_locs = torch.zeros(bs, dtype=torch.int64, device=device)
    if bs > 1:
        start_locs[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)

    for i in range(bs):
        src_start = start_locs[i].item()
        src_len = extend_seq_lens[i].item()
        dst_start = i * max_len

        padded_input_ids[dst_start:dst_start + src_len] = input_ids[src_start:src_start + src_len]
        padded_positions[dst_start:dst_start + src_len] = positions[src_start:src_start + src_len]
        padded_out_cache_loc[dst_start:dst_start + src_len] = out_cache_loc[src_start:src_start + src_len]
        pad_mask[dst_start:dst_start + src_len] = True

    return padded_input_ids, padded_positions, padded_out_cache_loc, pad_mask, total_padded


def pad_batch_to_uniform_gpu(
    input_ids: torch.Tensor,        # [total_tokens] flat
    positions: torch.Tensor,        # [total_tokens] flat
    out_cache_loc: torch.Tensor,    # [total_tokens] flat
    extend_seq_lens: torch.Tensor,  # [bs] per-request token counts
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-only padding using scatter operations (no CPU loop).

    Returns:
        padded_input_ids: [bs * max_len]
        padded_positions: [bs * max_len]
        padded_out_cache_loc: [bs * max_len]
        pad_mask: [bs * max_len]
    """
    device = input_ids.device
    bs = extend_seq_lens.shape[0]
    total_padded = bs * max_len

    # Compute source and destination indices
    # src_indices: [total_tokens] = [0, 1, ..., total_tokens-1]
    # dst_indices: maps each src token to its padded position
    start_locs = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    start_locs[1:] = torch.cumsum(extend_seq_lens, dim=0)
    total_tokens = start_locs[-1].item()

    # For each token, find which request it belongs to and its offset
    req_ids = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    for i in range(bs):
        req_ids[start_locs[i]:start_locs[i+1]] = i
    offsets = torch.arange(total_tokens, device=device) - start_locs[req_ids]
    dst_indices = req_ids * max_len + offsets

    # Allocate and scatter
    padded_input_ids = torch.full((total_padded,), PAD_TOKEN_ID, dtype=input_ids.dtype, device=device)
    padded_positions = torch.zeros(total_padded, dtype=positions.dtype, device=device)
    padded_out_cache_loc = torch.full((total_padded,), PADDING_SLOT_ID, dtype=out_cache_loc.dtype, device=device)

    padded_input_ids.scatter_(0, dst_indices, input_ids[:total_tokens])
    padded_positions.scatter_(0, dst_indices, positions[:total_tokens])
    padded_out_cache_loc.scatter_(0, dst_indices.to(out_cache_loc.dtype), out_cache_loc[:total_tokens])

    pad_mask = torch.zeros(total_padded, dtype=torch.bool, device=device)
    pad_mask.scatter_(0, dst_indices, torch.ones(total_tokens, dtype=torch.bool, device=device))

    return padded_input_ids, padded_positions, padded_out_cache_loc, pad_mask


def unpad_output(
    output: torch.Tensor,  # [bs * max_len, ...] padded output
    pad_mask: torch.Tensor,  # [bs * max_len] bool
) -> torch.Tensor:
    """Extract valid (non-padded) elements from output."""
    return output[pad_mask]


def unpad_output_per_req(
    output: torch.Tensor,  # [bs * max_len, ...] padded output
    extend_seq_lens: torch.Tensor,  # [bs] actual lengths
    max_len: int,
) -> torch.Tensor:
    """Extract last valid token per request (for sampling)."""
    bs = extend_seq_lens.shape[0]
    # Index of last valid token per request
    last_indices = torch.arange(bs, device=output.device) * max_len + extend_seq_lens - 1
    return output[last_indices]
