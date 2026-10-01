import math
from typing import Optional, Tuple

import torch
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# =============================================================================
# Base SDPA (used for prefill in all modes)
# =============================================================================

def sdpa_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, query.shape[-3] // key.shape[-3])
        value = repeat_kv(value, query.shape[-3] // value.shape[-3])

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def register_sdpa():
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward


# =============================================================================
# Full einsum decode attention
# =============================================================================

def full_einsum_forward(query, key, value, attention_mask=None, is_causal=False, scaling=None):
    b, q_h, l_q, d = query.shape
    k_h, v_h = key.shape[1], value.shape[1]
    l_kv = key.shape[2]
    scaling = 1 / math.sqrt(query.size(-1)) if scaling is None else scaling

    query_bmm = query.reshape(b * k_h, (q_h // k_h) * l_q, d)
    key_bmm = key.reshape(b * k_h, l_kv, d).transpose(-2, -1)
    attn_weights = torch.bmm(query_bmm, key_bmm)
    attn_weights = attn_weights.reshape(b, q_h, l_q, l_kv) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_bmm = attn_weights.reshape(b * v_h, (q_h // v_h) * l_q, l_kv)
    value_bmm = value.reshape(b * v_h, l_kv, d)
    attn_output = torch.bmm(attn_bmm, value_bmm)

    return attn_output.reshape(b, q_h, l_q, d)


# =============================================================================
# Atop-N amortized attention
# =============================================================================

MAX_LAYERS = 128
AMORTIZED_THRESHOLD_TENSOR: Optional[torch.Tensor] = None
EXEMPT_LAYERS: set = set()


def init_amortized_threshold_tensor(device, num_layers, num_rows):
    global AMORTIZED_THRESHOLD_TENSOR
    AMORTIZED_THRESHOLD_TENSOR = torch.zeros(num_layers, num_rows, device=device, dtype=torch.float32)


def reset_amortized_state():
    global AMORTIZED_THRESHOLD_TENSOR
    if AMORTIZED_THRESHOLD_TENSOR is not None:
        AMORTIZED_THRESHOLD_TENSOR.zero_()


def _threshold_topk_triton(x: torch.Tensor, k: int, threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from attention.triton_atopn import threshold_topk_fast
    return threshold_topk_fast(x, k, threshold)


def _compute_attn_weights(query, key, scaling, attention_mask):
    b, q_h, l_q, d = query.shape
    k_h = key.shape[1]
    l_kv = key.shape[2]
    scaling = 1 / math.sqrt(query.size(-1)) if scaling is None else scaling

    query_bmm = query.reshape(b * k_h, (q_h // k_h) * l_q, d)
    key_bmm = key.reshape(b * k_h, l_kv, d).transpose(-2, -1)
    attn_weights = torch.bmm(query_bmm, key_bmm)
    attn_weights = attn_weights.reshape(b, q_h, l_q, l_kv) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    return attn_weights


def _compute_attn_probs(query, key, scaling, attention_mask):
    attn_weights = _compute_attn_weights(query, key, scaling, attention_mask)
    return torch.nn.functional.softmax(attn_weights, dim=-1)


def _transform_softmax_threshold(attn_logits, softmax_threshold, tail_size=256):
    """Transform softmax-space threshold to logit-space via approximate inverse softmax."""
    nr = attn_logits.shape[0] * attn_logits.shape[1]
    l_kv = attn_logits.shape[-1]
    logits_2d = attn_logits.reshape(nr, l_kv)

    tail = min(tail_size, l_kv)
    m_approx = logits_2d[:, -tail:].max(dim=-1).values

    s = (logits_2d - m_approx.unsqueeze(1)).exp().sum(dim=-1)

    logit_threshold = m_approx + torch.log(softmax_threshold.clamp(min=1e-30) * s)

    return logit_threshold


def _atopn_gather_and_output(attn_weights_topk, attn_indices, value, query_shape, top_n_attention,
                              input_is_logits=False):
    b, q_h, l_q, d = query_shape
    v_h = value.shape[1]
    buf_k = attn_weights_topk.shape[-1]

    if input_is_logits:
        attn_probs_topk = torch.nn.functional.softmax(attn_weights_topk, dim=-1)
    else:
        attn_probs_topk = attn_weights_topk / attn_weights_topk.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    attn_indices = attn_indices[..., None].reshape(b, v_h, q_h//v_h, buf_k * l_q, 1).expand(-1, -1, -1, -1, value.shape[-1])
    value = value[:, :, None].expand(-1, -1, q_h // v_h, -1, -1).gather(dim=-2, index=attn_indices).reshape(attn_probs_topk.shape + value.shape[-1:])

    attn_bmm = attn_probs_topk.reshape(b * q_h * l_q, 1, buf_k * l_q)
    value_bmm = value.reshape(b * q_h * l_q, buf_k * l_q, d)
    attn_output = torch.bmm(attn_bmm, value_bmm)

    return attn_output.reshape(b, q_h, l_q, d)


def approx_top_n_attention_amortized_cal(query, key, value, attention_mask=None, is_causal=False,
                                          top_n_attention: int = 256, scaling=None, layer_idx: int = 0):
    """First decode step: full softmax+topk, stores threshold in AMORTIZED_THRESHOLD_TENSOR."""
    b, q_h, l_q, d = query.shape
    l_kv = key.shape[2]

    attn_probs = _compute_attn_probs(query, key, scaling, attention_mask)

    k = min(top_n_attention, l_kv)
    attn_probs_topk, attn_indices = torch.topk(attn_probs, k, dim=-1)

    num_rows = b * q_h
    threshold = attn_probs_topk[:, :, 0, -1].reshape(num_rows)

    AMORTIZED_THRESHOLD_TENSOR[layer_idx, :num_rows] = threshold

    return _atopn_gather_and_output(attn_probs_topk, attn_indices, value, query.shape, k)


def approx_top_n_attention_amortized(query, key, value, attention_mask=None, is_causal=False,
                                      top_n_attention: int = 256, scaling=None, layer_idx: int = 0):
    """Subsequent decode steps: logit-space threshold using cached thresholds + Triton kernel."""
    b, q_h, l_q, d = query.shape
    l_kv = key.shape[2]

    attn_logits = _compute_attn_weights(query, key, scaling, attention_mask)

    num_rows = b * q_h
    softmax_threshold = AMORTIZED_THRESHOLD_TENSOR[layer_idx, :num_rows]

    logit_threshold = _transform_softmax_threshold(attn_logits, softmax_threshold)

    buf_k = min(top_n_attention * 2, l_kv)
    attn_logits_topk, attn_indices = _threshold_topk_triton(attn_logits, buf_k, logit_threshold)

    return _atopn_gather_and_output(attn_logits_topk, attn_indices, value, query.shape,
                                     top_n_attention, input_is_logits=True)
