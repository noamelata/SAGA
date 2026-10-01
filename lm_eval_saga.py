import re
import sys
import argparse
import torch
from typing import Optional, Tuple
from lm_eval.models.huggingface import HFLM
from lm_eval.__main__ import cli_evaluate
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from attention.attention import (
    sdpa_attention_forward as _base_sdpa_forward,
    full_einsum_forward,
    approx_top_n_attention_amortized_cal,
    approx_top_n_attention_amortized,
    reset_amortized_state,
    init_amortized_threshold_tensor,
)
import attention.attention as _attn_module
from speedup import merge_key_projection_heads

# =============================================================================
# Globals set by CLI args
# =============================================================================

NUM_KEY_HEADS = None   # --num_key_heads
ATOPN_VALUE = None     # --atopn

# =============================================================================
# Per-model default exempt layers for amortized attention.
# These layers have attention patterns where the tail-based m_approx in
# _transform_softmax_threshold underestimates the true max, causing exp
# overflow → inf threshold → NaN output.  Exempt layers use full attention.
# Keys are matched as substrings against the model name/path (case-insensitive).
# =============================================================================

MODEL_EXEMPT_LAYERS = {
    "qwen2.5-1.5b": {0, 14},
    "llama-3.2-1b": {0},
}

# =============================================================================
# Per-step calibration counter for amortized mode
# =============================================================================

_AMORTIZED_LAYER0_COUNT = 0  # counts decode steps since last prefill

# =============================================================================
# Post-load merge via HFLM._create_model patch
# =============================================================================

_original_create_model = HFLM._create_model

def _patched_create_model(self, *args, **kwargs):
    # Check if the model checkpoint was saved with already-merged KV heads
    pretrained = kwargs.get("pretrained", args[0] if args else None)
    _already_merged = False
    _already_expanded_v = False
    _ckpt_num_key_heads = None
    _ckpt_num_value_heads = None
    if pretrained is not None:
        import json, os
        config_path = os.path.join(pretrained, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            _ckpt_num_key_heads = cfg.get("num_key_heads")
            if _ckpt_num_key_heads is not None:
                _already_merged = True
            _ckpt_num_value_heads = cfg.get("num_value_heads")
            if _ckpt_num_value_heads is not None:
                _already_expanded_v = True

    if _already_merged or _already_expanded_v:
        # Temporarily patch from_pretrained to allow size mismatches on k_proj/v_proj
        from transformers import AutoModelForCausalLM
        _orig_fp = AutoModelForCausalLM.from_pretrained.__func__
        @classmethod
        def _ims_from_pretrained(cls, *a, **kw):
            kw["ignore_mismatched_sizes"] = True
            return _orig_fp(cls, *a, **kw)
        AutoModelForCausalLM.from_pretrained = _ims_from_pretrained
        _original_create_model(self, *args, **kwargs)
        AutoModelForCausalLM.from_pretrained = classmethod(_orig_fp)

        # Now fix k_proj: the model has default-sized k_proj (reinit'd),
        # but checkpoint has merged (smaller) k_proj. Load them manually.
        import torch.nn as nn
        from safetensors.torch import load_file
        ckpt_path = os.path.join(pretrained, "model.safetensors")
        if os.path.exists(ckpt_path):
            ckpt = load_file(ckpt_path, device="cpu")
        else:
            import glob as globmod
            shard_files = sorted(globmod.glob(os.path.join(pretrained, "model-*.safetensors")))
            ckpt = {}
            for sf in shard_files:
                ckpt.update(load_file(sf, device="cpu"))

        model_cfg = self._model.config
        head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads
        for layer_idx, layer in enumerate(self._model.model.layers):
            w_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            b_key = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            if w_key in ckpt:
                w = ckpt[w_key]
                has_bias = b_key in ckpt
                new_k = nn.Linear(w.shape[1], w.shape[0], bias=has_bias,
                                  device=layer.self_attn.k_proj.weight.device,
                                  dtype=layer.self_attn.k_proj.weight.dtype)
                new_k.weight = nn.Parameter(w.to(device=new_k.weight.device, dtype=new_k.weight.dtype))
                if has_bias:
                    new_k.bias = nn.Parameter(ckpt[b_key].to(device=new_k.bias.device, dtype=new_k.bias.dtype))
                layer.self_attn.k_proj = new_k
            # Fix v_proj if checkpoint has expanded V heads
            if _already_expanded_v:
                vw_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                vb_key = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
                if vw_key in ckpt:
                    vw = ckpt[vw_key]
                    v_has_bias = vb_key in ckpt
                    new_v = nn.Linear(vw.shape[1], vw.shape[0], bias=v_has_bias,
                                      device=layer.self_attn.v_proj.weight.device,
                                      dtype=layer.self_attn.v_proj.weight.dtype)
                    new_v.weight = nn.Parameter(vw.to(device=new_v.weight.device, dtype=new_v.weight.dtype))
                    if v_has_bias:
                        new_v.bias = nn.Parameter(ckpt[vb_key].to(device=new_v.bias.device, dtype=new_v.bias.dtype))
                    layer.self_attn.v_proj = new_v
        del ckpt
        msg_parts = []
        if _already_merged:
            msg_parts.append(f"num_key_heads={_ckpt_num_key_heads}")
        if _already_expanded_v:
            msg_parts.append(f"num_value_heads={_ckpt_num_value_heads}")
        print(f"Loaded pre-merged checkpoint ({', '.join(msg_parts)})")
    else:
        _original_create_model(self, *args, **kwargs)
        if NUM_KEY_HEADS is not None:
            merge_key_projection_heads(self._model, new_heads=NUM_KEY_HEADS)

    # Clear max_new_tokens from generation_config so it doesn't override
    # lm_eval's max_length parameter in model.generate()
    if hasattr(self._model, 'generation_config') and self._model.generation_config.max_new_tokens is not None:
        print(f"Clearing generation_config.max_new_tokens={self._model.generation_config.max_new_tokens} "
              f"(lm_eval controls generation length via max_length)")
        self._model.generation_config.max_new_tokens = None

HFLM._create_model = _patched_create_model

# =============================================================================
# Phase-aware attention: SDPA for prefill, einsum/atopn for decode
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
    l_q = query.shape[2]

    if l_q == 1:
        # Decode phase — dispatch based on available thresholds
        global _AMORTIZED_LAYER0_COUNT
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key.shape[-2]]

        if ATOPN_VALUE is not None:
            layer_idx = getattr(module, 'layer_idx', 0) if module is not None else 0

            # Counter at layer 0 ensures all layers in the same step see the same value
            if layer_idx == 0:
                _AMORTIZED_LAYER0_COUNT += 1
                if _attn_module.AMORTIZED_THRESHOLD_TENSOR is None:
                    num_rows = query.shape[0] * query.shape[1]
                    init_amortized_threshold_tensor(query.device, _attn_module.MAX_LAYERS, num_rows)

            if layer_idx in _attn_module.EXEMPT_LAYERS:
                attn_output = full_einsum_forward(
                    query, key, value, causal_mask, scaling=scaling,
                )
            elif _AMORTIZED_LAYER0_COUNT <= 1 or _AMORTIZED_LAYER0_COUNT % 32 == 1:
                # Recalibrate thresholds on first step and every 32 steps
                attn_output = approx_top_n_attention_amortized_cal(
                    query, key, value, causal_mask,
                    scaling=scaling, top_n_attention=ATOPN_VALUE, layer_idx=layer_idx,
                )
            else:
                attn_output = approx_top_n_attention_amortized(
                    query, key, value, causal_mask,
                    scaling=scaling, top_n_attention=ATOPN_VALUE, layer_idx=layer_idx,
                )
        else:
            attn_output = full_einsum_forward(
                query, key, value, causal_mask, scaling=scaling,
            )

        return attn_output.transpose(1, 2).contiguous(), None

    # Prefill phase — reset amortized state and use base SDPA
    if ATOPN_VALUE is not None:
        layer_idx = getattr(module, 'layer_idx', 0) if module is not None else 0
        if layer_idx == 0:
            _AMORTIZED_LAYER0_COUNT = 0
            reset_amortized_state()

    return _base_sdpa_forward(module, query, key, value, attention_mask,
                              dropout=dropout, scaling=scaling, is_causal=is_causal)

ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_key_heads', type=int, default=None)
    parser.add_argument('--atopn', type=int, default=None)
    parser.add_argument('--exempt_layers', type=str, default='',
                        help='Comma-separated layer indices to exempt from approximation (use full attention)')
    our_args, remaining = parser.parse_known_args()

    NUM_KEY_HEADS = our_args.num_key_heads
    ATOPN_VALUE = our_args.atopn

    if our_args.exempt_layers:
        _attn_module.EXEMPT_LAYERS = {int(x.strip()) for x in our_args.exempt_layers.split(',')}
        print(f"Exempt layers (full attention): {sorted(_attn_module.EXEMPT_LAYERS)}")
    elif ATOPN_VALUE is not None:
        # Auto-detect exempt layers from model name in remaining args
        model_name = ''
        for i, arg in enumerate(remaining):
            if arg == '--model_args' and i + 1 < len(remaining):
                model_name = remaining[i + 1].lower()
                break
        for pattern, layers in MODEL_EXEMPT_LAYERS.items():
            if pattern in model_name:
                _attn_module.EXEMPT_LAYERS = layers
                print(f"Auto-detected exempt layers for '{pattern}': {sorted(_attn_module.EXEMPT_LAYERS)}")
                break

    if ATOPN_VALUE is not None:
        print(f"Using amortized kthvalue thresholds (atopn={ATOPN_VALUE})")

    sys.argv = [sys.argv[0]] + remaining
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli_evaluate())
