import argparse
import sys
import time
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from typing import Optional, Tuple
import numpy as np
import math

# Allow layer_idx to be dynamic in torch.compile (avoid recompilation per layer)
import torch._dynamo.config
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from attention.attention import (
    sdpa_attention_forward as _base_sdpa_forward,
    full_einsum_forward,
    approx_top_n_attention_amortized_cal,
    approx_top_n_attention_amortized,
    init_amortized_threshold_tensor,
    reset_amortized_state,
    MAX_LAYERS,
)
import attention.attention as _attn_module

torch.set_float32_matmul_precision('high')


# =============================================================================
# Attention mode globals (set by CLI)
# =============================================================================

TOP_N_ATTENTION = None
ATTENTION_MODE = "vanilla"

# Step counter for atopn_amortized: counts decode steps since last prefill.
_AMORTIZED_STEP_COUNT = 0


def sdpa_attention_forward_dispatch(module, query, key, value, attention_mask=None, dropout=0.0,
                          scaling=None, is_causal=None, **kwargs) -> Tuple[torch.Tensor, None]:
    """Main attention dispatcher for benchmarking."""
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key.shape[-2]].to(query.device)

    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # Dispatch based on attention mode (only for decode steps with l_q=1)
    if query.shape[-2] == 1 and ATTENTION_MODE == "full":
        attn_output = full_einsum_forward(query, key, value, causal_mask, is_causal, scaling)
    elif query.shape[-2] == 1 and ATTENTION_MODE == "atopn":
        global _AMORTIZED_STEP_COUNT
        layer_idx = getattr(module, 'layer_idx', 0) if module is not None else 0
        # Counter at layer 0 ensures all layers in the same step see the same value
        if layer_idx == 0:
            _AMORTIZED_STEP_COUNT += 1
        if layer_idx in _attn_module.EXEMPT_LAYERS:
            attn_output = full_einsum_forward(query, key, value, causal_mask, is_causal, scaling)
        elif _AMORTIZED_STEP_COUNT <= 1:
            # First decode step: calibrate thresholds via full softmax + topk
            attn_output = approx_top_n_attention_amortized_cal(query, key, value, causal_mask,
                                                                scaling=scaling, top_n_attention=TOP_N_ATTENTION,
                                                                layer_idx=layer_idx)
        else:
            attn_output = approx_top_n_attention_amortized(query, key, value, causal_mask,
                                                            scaling=scaling, top_n_attention=TOP_N_ATTENTION,
                                                            layer_idx=layer_idx)
    else:  # vanilla or prefill
        # Reset amortized step counter on prefill so next generation recalibrates
        if query.shape[-2] > 1:
            layer_idx = getattr(module, 'layer_idx', 0) if module is not None else 0
            if layer_idx == 0:
                _AMORTIZED_STEP_COUNT = 0
        return _base_sdpa_forward(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling, is_causal=is_causal)

    return attn_output.transpose(1, 2).contiguous(), None


# =============================================================================
# Model Setup
# =============================================================================

def merge_key_projection_heads(model, new_heads=1):
    """Merge key projection heads using average method."""
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    for layer in model.model.layers:
        k_proj = layer.self_attn.k_proj
        new_proj = k_proj.weight.reshape(new_heads, -1, head_dim, hidden_size).mean(1).reshape(
            new_heads * head_dim, hidden_size)

        has_bias = k_proj.bias is not None
        new_bias = k_proj.bias.reshape(-1, head_dim).mean(0) if has_bias else None

        new_k_proj = torch.nn.Linear(hidden_size, new_heads * head_dim, bias=has_bias,
                                     device=k_proj.weight.device, dtype=k_proj.weight.dtype)
        new_k_proj.weight = torch.nn.Parameter(new_proj.to(k_proj.weight), requires_grad=False)
        if has_bias:
            new_k_proj.bias = torch.nn.Parameter(new_bias.to(k_proj.bias), requires_grad=False)

        layer.self_attn.k_proj = new_k_proj

    print(f"Merged k_proj heads (new_heads={new_heads})")
    return model


def initialize_kv_cache(model, max_cache_len, num_key_heads=None, device="cuda", dtype=torch.float32):
    """Initialize StaticCache with potentially different key and value head counts."""
    past_kv = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len,
                          device=device, dtype=dtype)

    if num_key_heads is not None:
        num_value_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        for layer in past_kv.layers:
            layer.keys = torch.zeros((1, num_key_heads, max_cache_len, head_dim), dtype=dtype, device=device)
            layer.values = torch.zeros((1, num_value_heads, max_cache_len, head_dim), dtype=dtype, device=device)
            layer.is_initialized = True
            layer.dtype = dtype
            layer.device = device
            layer.max_batch_size = 1
            layer.num_heads = num_key_heads
            layer.k_head_dim = head_dim
            layer.v_head_dim = head_dim

    return past_kv


# =============================================================================
# Benchmark Functions
# =============================================================================

def prefill(model, input_ids, new_heads=None, differing_batches=False, dtype=torch.float32, num_new_tokens=8):
    """Run prefill and return KV cache."""
    max_cache_len = input_ids.shape[-1] + num_new_tokens
    past_kv = initialize_kv_cache(model, max_cache_len, num_key_heads=new_heads,
                                  device=input_ids.device, dtype=dtype)
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:1]
    outputs = model(input_ids, use_cache=True, logits_to_keep=1, past_key_values=past_kv)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    next_token = next_token.repeat((batch_size, 1))
    for layer in past_kv.layers:
        layer.max_batch_size = batch_size
        layer.keys = layer.keys.repeat((batch_size, 1, 1, 1))
        layer.values = layer.values.repeat((batch_size, 1, 1, 1))
    return past_kv, next_token


def decode(model, past_kv, next_token, num_new_tokens):
    """Run decode steps."""
    for _ in range(num_new_tokens):
        outputs = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)


# Keep an uncompiled reference for calibration warmup
_decode_uncompiled = decode

# Global profiler reference for decode-only profiling
_DECODE_PROFILER = None


def decode_with_graph(model, past_kv, next_token, num_new_tokens,
                      graph_cal, graph_fast, static_input, static_logits):
    """Run decode steps by replaying captured CUDA graphs.

    For atopn_amortized mode, graph_cal is replayed on step 0 (calibration path)
    and graph_fast on steps 1+ (cached-threshold path). For other modes,
    graph_cal is None and graph_fast is always replayed.
    """
    for i in range(num_new_tokens):
        static_input.copy_(next_token)
        if i == 0 and graph_cal is not None:
            graph_cal.replay()
        else:
            graph_fast.replay()
        next_token = static_logits.argmax(dim=-1)



def capture_cuda_graphs(model, past_kv, next_token, static_input, static_logits,
                        requested_mode, num_warmup=3):
    """Capture CUDA graph(s) for the decode step.

    Returns (graph_cal, graph_fast, graph_deps). graph_cal is None for non-amortized modes.
    graph_deps holds tensors that must stay alive while graphs are in use.
    """
    global _AMORTIZED_STEP_COUNT, ATTENTION_MODE

    graph_cal = None
    graph_fast = None

    # Pre-compute inputs that would normally be dynamically created during forward
    device = next_token.device
    dtype = next(model.parameters()).dtype
    seen_tokens = past_kv.get_seq_length()
    cache_position = torch.arange(seen_tokens, seen_tokens + 1, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)

    # Create the causal mask
    batch_size = past_kv.layers[0].keys.shape[0]
    inputs_embeds = model.model.embed_tokens(next_token[:1])
    inputs_embeds = inputs_embeds.expand(batch_size, -1, -1)
    from transformers.masking_utils import create_causal_mask
    causal_mask = create_causal_mask(
        config=model.config, inputs_embeds=inputs_embeds, attention_mask=None,
        cache_position=cache_position, past_key_values=past_kv, position_ids=position_ids,
    )
    causal_mask_dict = {"full_attention": causal_mask}

    def _reset_cache():
        for layer in past_kv.layers:
            layer.keys[:, :, seen_tokens:, :] = 0
            layer.values[:, :, seen_tokens:, :] = 0

    def _run_forward():
        static_input.copy_(next_token)
        out = model(static_input, past_key_values=past_kv, use_cache=True,
                    cache_position=cache_position, position_ids=position_ids,
                    attention_mask=causal_mask_dict)
        static_logits.copy_(out.logits[:, -1:, :])
        del out

    # Warmup: run the model forward multiple times OUTSIDE graph capture to
    # trigger all lazy initializations, CUDA memory allocations, and — when
    # model.forward is compiled — allow torch.compile to finish compilation.
    ATTENTION_MODE = requested_mode
    if requested_mode == "atopn":
        for _ in range(num_warmup):
            # Warmup calibration path (step 0)
            _AMORTIZED_STEP_COUNT = 0
            _run_forward()
            _reset_cache()
            # Warmup fast path (step 1+)
            _run_forward()
            _reset_cache()
    else:
        _AMORTIZED_STEP_COUNT = 2
        for _ in range(num_warmup):
            _run_forward()
            _reset_cache()
    torch.cuda.synchronize()

    # Capture CUDA graphs
    if requested_mode == "atopn":
        # Capture calibration graph (step 0 path)
        _AMORTIZED_STEP_COUNT = 0
        graph_cal = torch.cuda.CUDAGraph()
        static_input.copy_(next_token)
        with torch.cuda.graph(graph_cal):
            out = model(static_input, past_key_values=past_kv, use_cache=True,
                        cache_position=cache_position, position_ids=position_ids,
                        attention_mask=causal_mask_dict)
            static_logits.copy_(out.logits[:, -1:, :])

        # Capture fast graph (step 1+ path)
        graph_fast = torch.cuda.CUDAGraph()
        static_input.copy_(next_token)
        with torch.cuda.graph(graph_fast):
            out = model(static_input, past_key_values=past_kv, use_cache=True,
                        cache_position=cache_position, position_ids=position_ids,
                        attention_mask=causal_mask_dict)
            static_logits.copy_(out.logits[:, -1:, :])
    else:
        graph_fast = torch.cuda.CUDAGraph()
        static_input.copy_(next_token)
        with torch.cuda.graph(graph_fast):
            out = model(static_input, past_key_values=past_kv, use_cache=True,
                        cache_position=cache_position, position_ids=position_ids,
                        attention_mask=causal_mask_dict)
            static_logits.copy_(out.logits[:, -1:, :])

    # Return tensors that must stay alive while graphs are in use
    graph_deps = [cache_position, position_ids, causal_mask, causal_mask_dict, inputs_embeds]
    return graph_cal, graph_fast, graph_deps

def measure_decode_latency(model, input_ids, num_new_tokens=1, num_warmup=5, num_iterations=100,
                           new_heads=None, dtype=torch.float32, calibrate_during_measurement=False,
                           profile_decode_only=False, use_cuda_graph=False):
    """Measure decode-only latency."""
    global _DECODE_PROFILER, ATTENTION_MODE, _AMORTIZED_STEP_COUNT
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    requested_mode = ATTENTION_MODE

    # Pre-allocate AMORTIZED_THRESHOLD_TENSOR before warmup/compilation
    if requested_mode == "atopn":
        with torch.no_grad():
            past_kv, next_token = prefill(model, input_ids, new_heads, dtype=dtype, num_new_tokens=num_new_tokens)
            num_rows = next_token.shape[0] * model.config.num_attention_heads
            init_amortized_threshold_tensor(device, MAX_LAYERS, num_rows)

    # Phase 2: Compile warmup (compiled decode)
    # When using CUDA graphs, model.forward is compiled instead of decode(),
    # and the compile warmup happens inside capture_cuda_graphs(). Skip here
    # to avoid holding large KV caches in memory unnecessarily.
    if not use_cuda_graph:
        with torch.no_grad():
            for _ in range(num_warmup):
                if requested_mode == "atopn":
                    _AMORTIZED_STEP_COUNT = 0
                    reset_amortized_state()
                past_kv, next_token = prefill(model, input_ids, new_heads, dtype=dtype, num_new_tokens=num_new_tokens)
                ATTENTION_MODE = requested_mode
                decode(model, past_kv, next_token, num_new_tokens)
        del past_kv, next_token

    # Phase 2.5: CUDA graph capture (after warmup, before measurement)
    graph_cal = None
    graph_fast = None
    graph_deps = None
    static_input = None
    static_logits = None
    capture_past_kv = None
    capture_next_token = None

    if use_cuda_graph:
        torch.cuda.synchronize()
        # Force GC and free cached CUDA memory from warmup to avoid address
        # conflicts during graph capture. Without this, the CUDA caching
        # allocator may reuse memory from freed warmup caches, causing the
        # graph's index_copy_ to hit stale addresses during replay.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            # Get a fresh prefill state for graph capture
            if requested_mode == "atopn":
                _AMORTIZED_STEP_COUNT = 0
                reset_amortized_state()
            past_kv, next_token = prefill(model, input_ids, new_heads, dtype=dtype, num_new_tokens=num_new_tokens)
            ATTENTION_MODE = requested_mode

            batch_size = next_token.shape[0]
            vocab_size = model.config.vocab_size

            # Allocate static buffers at fixed addresses for graph replay
            static_input = torch.empty_like(next_token)  # (batch_size, 1)
            static_logits = torch.empty(batch_size, 1, vocab_size, device=device, dtype=dtype)

            # Mark StaticCache tensors as having static addresses
            for layer in past_kv.layers:
                torch._dynamo.mark_static_address(layer.keys)
                torch._dynamo.mark_static_address(layer.values)
            if _attn_module.AMORTIZED_THRESHOLD_TENSOR is not None:
                torch._dynamo.mark_static_address(_attn_module.AMORTIZED_THRESHOLD_TENSOR)

            graph_cal, graph_fast, graph_deps = capture_cuda_graphs(
                model, past_kv, next_token, static_input, static_logits,
                requested_mode)

            # Save capture state for reuse during measurement
            capture_past_kv = past_kv
            capture_next_token = next_token

    torch.cuda.synchronize()
    latencies = []

    # Phase 3: Measurement
    with torch.no_grad():
        for _ in range(num_iterations):
            if use_cuda_graph:
                # CUDA graph replay: reuse the captured past_kv (same tensor addresses).
                # The graph always writes to the same cache_position via index_copy_,
                # so no cache state reset is needed.
                if requested_mode == "atopn":
                    _AMORTIZED_STEP_COUNT = 0
                    reset_amortized_state()
                next_token = capture_next_token

                torch.cuda.synchronize()

                if profile_decode_only and _DECODE_PROFILER is not None:
                    _DECODE_PROFILER.start()

                start = time.perf_counter()
                decode_with_graph(model, capture_past_kv, next_token, num_new_tokens,
                                  graph_cal, graph_fast, static_input, static_logits)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

                if profile_decode_only and _DECODE_PROFILER is not None:
                    _DECODE_PROFILER.stop()
            else:
                if requested_mode == "atopn":
                    _AMORTIZED_STEP_COUNT = 0
                    reset_amortized_state()
                past_kv, next_token = prefill(model, input_ids, new_heads, dtype=dtype, num_new_tokens=num_new_tokens)
                ATTENTION_MODE = requested_mode

                torch.cuda.synchronize()

                if profile_decode_only and _DECODE_PROFILER is not None:
                    _DECODE_PROFILER.start()

                start = time.perf_counter()
                decode(model, past_kv, next_token, num_new_tokens)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

                if profile_decode_only and _DECODE_PROFILER is not None:
                    _DECODE_PROFILER.stop()

    # Restore mode
    ATTENTION_MODE = requested_mode

    # Free CUDA graph state to release GPU memory for subsequent runs
    del graph_cal, graph_fast, graph_deps, static_input, static_logits
    del capture_past_kv, capture_next_token
    # Clear Triton buffer cache (holds GPU tensors from threshold_topk_fast)
    from attention.triton_atopn import _BUFFER_CACHE
    _BUFFER_CACHE.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # All modes now measure the same number of tokens
    latencies = np.array(latencies) / num_new_tokens
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p90_ms": np.percentile(latencies, 90),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure decode latency with various attention modes")
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--seq_lengths", type=str, default="65536")
    parser.add_argument("--batch_sizes", type=str, default="32")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--new_heads", type=int, default=1)
    parser.add_argument("--num_new_tokens", type=int, default=8)
    parser.add_argument("--merge_mode", type=str, default="merge")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--attention", type=str, default="full",
                        help="Attention modes: 'vanilla', 'full', 'atopn:N', 'atopn_amortized:N'")
    parser.add_argument("--calibrate_during_measurement", action="store_true",
                        help="Keep calibration on during measurement iterations. "
                             "Includes calibration cost and causes graph breaks with torch.compile.")
    parser.add_argument("--profile_decode", action="store_true",
                        help="Profile only decode phase (excludes warmup and prefill from trace)")
    parser.add_argument("--profile_output", type=str, default="decode_trace.json",
                        help="Output filename for profiler trace (default: decode_trace.json)")
    parser.add_argument("--exempt_layers", type=str, default="",
                        help="Comma-separated layer indices to exempt from approximation (use full attention)")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--cuda_graph", action="store_true",
                        help="Use manual CUDA graph capture for decode steps to reduce CPU dispatch overhead")
    parser.add_argument("--_subprocess_json", action="store_true",
                        help=argparse.SUPPRESS)  # Internal: subprocess outputs JSON result
    args = parser.parse_args()

    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    merge_modes = [m.strip().lower() for m in args.merge_mode.split(",")]

    # Parse attention modes
    attention_modes = []
    for val in args.attention.split(","):
        val = val.strip().lower()
        if val in ("vanilla", "full"):
            attention_modes.append((val, None))
        elif val.startswith("atopn:"):
            attention_modes.append(("atopn", int(val.split(":")[1])))
        elif val.startswith("atopn_amortized:"):
            attention_modes.append(("atopn", int(val.split(":")[1])))
        else:
            raise ValueError(f"Unknown attention mode: {val}")

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward_dispatch

    global decode
    if args.exempt_layers:
        _attn_module.EXEMPT_LAYERS = {int(x.strip()) for x in args.exempt_layers.split(",")}
    if not args.no_compile and not args.cuda_graph:
        # Compile the decode() wrapper for non-CUDA-graph path.
        # For CUDA graph path, we compile model.forward after loading (below).
        if args.compile_mode == "default":
            decode = torch.compile(decode)
        else:
            decode = torch.compile(decode, mode=args.compile_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if not args._subprocess_json:
        print(f"\nConfig: dtype={dtype}, batch_sizes={batch_sizes}, seq_lengths={seq_lengths}")
        print(f"Attention modes: {[f'{m}:{n}' if n else m for m, n in attention_modes]}")
        if _attn_module.EXEMPT_LAYERS:
            print(f"Exempt layers (full attention): {sorted(_attn_module.EXEMPT_LAYERS)}")
        compile_status = 'no' if args.no_compile else f'yes (mode={args.compile_mode})'
        print(f"Compile: {compile_status}")
        if args.cuda_graph:
            if args.no_compile:
                print(f"CUDA graph: enabled (uncompiled)")
            else:
                print(f"CUDA graph: enabled (compiled + graph capture)")
        if args.calibrate_during_measurement:
            print(f"Calibration: during measurement (graph breaks expected)")
        if args.profile_decode:
            print(f"Profiling decode: output={args.profile_output}")
        print("=" * 70)

    results = {}
    failures = []  # (merge, attention, batch, seq) configs that crashed or timed out (OOM is not a failure)
    use_subprocess = args.cuda_graph and not args.no_compile and not args._subprocess_json
    for merge_mode in merge_modes:
        do_merge = (merge_mode == "merge")
        merge_label = "merged" if do_merge else "baseline"

        if use_subprocess:
            # Subprocess isolation: torch.compile caches accumulate on GPU and
            # cannot be freed without torch._dynamo.reset(), which segfaults
            # when the compiled model is still referenced. Running each config
            # in a separate process ensures all GPU memory (compile caches,
            # CUDA graphs, KV caches) is fully reclaimed between runs.
            import subprocess, json, sys
            for mode, param in attention_modes:
                attn_label = f"{mode}:{param}" if param is not None else mode
                print(f"\n### {merge_label} / {attn_label} ###")
                for batch_size in batch_sizes:
                    for seq_len in seq_lengths:
                        # Build subprocess command with single config
                        sub_cmd = [
                            sys.executable, "speedup.py",
                            "--model_path", args.model_path,
                            "--attention", f"{mode}:{param}" if param else mode,
                            "--batch_sizes", str(batch_size),
                            "--seq_lengths", str(seq_len),
                            "--num_new_tokens", str(args.num_new_tokens),
                            "--num_iterations", str(args.num_iterations),
                            "--num_warmup", str(args.num_warmup),
                            "--dtype", args.dtype,
                            "--new_heads", str(args.new_heads),
                            "--merge_mode", merge_mode,
                            "--compile_mode", args.compile_mode,
                            "--cuda_graph",
                        ]
                        if args.exempt_layers:
                            sub_cmd += ["--exempt_layers", args.exempt_layers]
                        if args.calibrate_during_measurement:
                            sub_cmd.append("--calibrate_during_measurement")
                        if args.profile_decode:
                            sub_cmd += ["--profile_decode", "--profile_output", args.profile_output]
                        # Add sentinel flag so the subprocess outputs JSON
                        sub_cmd.append("--_subprocess_json")

                        try:
                            result = subprocess.run(
                                sub_cmd, capture_output=True, text=True, timeout=600,
                            )
                            if result.returncode == 0:
                                # Parse JSON output from subprocess
                                for line in result.stdout.strip().split("\n"):
                                    if line.startswith("{"):
                                        stats = json.loads(line)
                                        print(f"  BS={batch_size}, Seq={seq_len}: {stats['mean_ms']:.2f} ms (P50={stats['p50_ms']:.2f})")
                                        results[(merge_label, attn_label, batch_size, seq_len)] = stats
                                        break
                                else:
                                    # No JSON found — check for OOM in stdout or stderr
                                    combined = result.stdout + result.stderr
                                    if "OOM" in combined or "OutOfMemoryError" in combined:
                                        print(f"  BS={batch_size}, Seq={seq_len}: OOM")
                                    else:
                                        print(f"  BS={batch_size}, Seq={seq_len}: FAILED")
                                        failures.append((merge_label, attn_label, batch_size, seq_len))
                                        if result.stderr:
                                            err_lines = result.stderr.strip().split("\n")
                                            for line in err_lines[-3:]:
                                                print(f"    {line}")
                            else:
                                combined = result.stdout + result.stderr
                                if "OOM" in combined or "OutOfMemoryError" in combined:
                                    print(f"  BS={batch_size}, Seq={seq_len}: OOM")
                                else:
                                    print(f"  BS={batch_size}, Seq={seq_len}: FAILED (exit {result.returncode})")
                                    failures.append((merge_label, attn_label, batch_size, seq_len))
                                    err_lines = result.stderr.strip().split("\n")
                                    for line in err_lines[-3:]:
                                        print(f"    {line}")
                        except subprocess.TimeoutExpired:
                            print(f"  BS={batch_size}, Seq={seq_len}: TIMEOUT")
                            failures.append((merge_label, attn_label, batch_size, seq_len))
        else:
            if not args._subprocess_json:
                print(f"\nLoading model from {args.model_path}...")
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                                                         device_map="cuda:0")
            if do_merge:
                model = merge_key_projection_heads(model, new_heads=args.new_heads)
            model.eval()

            # Compile model.forward for CUDA graph path (without subprocess isolation)
            if args.cuda_graph and not args.no_compile:
                model.forward = torch.compile(model.forward, mode=args.compile_mode)

            effective_new_heads = args.new_heads if do_merge else None

            for mode, param in attention_modes:
                global TOP_N_ATTENTION, ATTENTION_MODE
                ATTENTION_MODE = mode
                TOP_N_ATTENTION = param if mode == "atopn" else None

                attn_label = f"{mode}:{param}" if param is not None else mode

                if not args._subprocess_json:
                    print(f"\n### {merge_label} / {attn_label} ###")

                for batch_size in batch_sizes:
                    for seq_len in seq_lengths:
                        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), dtype=torch.long)
                        try:
                            # Set up profiler for decode-only profiling
                            global _DECODE_PROFILER
                            if args.profile_decode:
                                _DECODE_PROFILER = profile(
                                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    record_shapes=True,
                                )

                            stats = measure_decode_latency(model, input_ids, args.num_new_tokens,
                                                          args.num_warmup, args.num_iterations,
                                                          effective_new_heads, dtype=dtype,
                                                          calibrate_during_measurement=args.calibrate_during_measurement,
                                                          profile_decode_only=args.profile_decode,
                                                          use_cuda_graph=args.cuda_graph)

                            # Export trace if profiling
                            if args.profile_decode and _DECODE_PROFILER is not None:
                                trace_name = f"{args.profile_output.replace('.json', '')}_{merge_label}_{attn_label.replace(':', '_')}_bs{batch_size}_seq{seq_len}.json"
                                _DECODE_PROFILER.export_chrome_trace(trace_name)
                                if not args._subprocess_json:
                                    print(f"  Trace saved to: {trace_name}")
                                _DECODE_PROFILER = None

                            if args._subprocess_json:
                                import json
                                print(json.dumps(stats))
                            else:
                                print(f"  BS={batch_size}, Seq={seq_len}: {stats['mean_ms']:.2f} ms (P50={stats['p50_ms']:.2f})")
                            results[(merge_label, attn_label, batch_size, seq_len)] = stats
                        except torch.cuda.OutOfMemoryError:
                            print(f"  BS={batch_size}, Seq={seq_len}: OOM")
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            # Clear Triton buffer cache to free GPU memory
                            from attention.triton_atopn import _BUFFER_CACHE
                            _BUFFER_CACHE.clear()

            del model
            torch.cuda.empty_cache()

    if not args._subprocess_json:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Merge':<10} {'Attention':<12} {'BS':<6} {'Seq':<8} {'Mean (ms)':<12} {'P50 (ms)':<12}")
        print("-" * 60)
        for (merge, attn, bs, seq), s in results.items():
            if s:
                print(f"{merge:<10} {attn:<12} {bs:<6} {seq:<8} {s['mean_ms']:<12.2f} {s['p50_ms']:<12.2f}")
        if failures:
            print(f"\n{len(failures)} configuration(s) FAILED: {failures}")

    return results, failures


if __name__ == "__main__":
    _, _failures = main()
    sys.exit(1 if _failures else 0)
