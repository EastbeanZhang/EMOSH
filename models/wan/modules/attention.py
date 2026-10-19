# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn.functional as F


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Attention for [batch, sequence, heads, channels] tensors."""
    output_dtype = q.dtype
    batch, lq, lk = q.shape[0], q.shape[1], k.shape[1]
    if q_lens is not None and bool(torch.all(q_lens == lq)):
        q_lens = None
    if k_lens is not None and bool(torch.all(k_lens == lk)):
        k_lens = None
    mask = None
    q_valid = None
    if q_lens is not None or k_lens is not None or causal or window_size != (-1, -1):
        q_len = torch.full((batch,), lq, device=q.device) if q_lens is None else q_lens.to(q.device)
        k_len = torch.full((batch,), lk, device=q.device) if k_lens is None else k_lens.to(q.device)
        qi = torch.arange(lq, device=q.device)[None, :, None]
        ki = torch.arange(lk, device=q.device)[None, None, :]
        q_valid = qi < q_len[:, None, None]
        mask = q_valid & (ki < k_len[:, None, None])
        center = qi + (k_len - q_len)[:, None, None]
        if causal:
            mask = mask & (ki <= center)
        left, right = window_size
        if left >= 0:
            mask = mask & (ki >= center - left)
        if right >= 0:
            mask = mask & (ki <= center + right)
        mask = mask[:, None]
    compute_dtype = dtype if q.device.type == "cuda" else output_dtype
    q, k, v = [x.transpose(1, 2).to(compute_dtype) for x in (q, k, v)]
    if q_scale is not None:
        q = q * q_scale
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=softmax_scale,
    ).transpose(1, 2)
    if q_valid is not None:
        output = output * q_valid.unsqueeze(-1)
    return output.to(output_dtype)


def attention(q, k, v, fa_version=None, **kwargs):
    return flash_attention(q, k, v, **kwargs)
