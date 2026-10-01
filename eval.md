# SAGA Evaluation Guide

All commands run from **inside `saga/`**. Set `WORKDIR` if models/experiments are not in `.`.

## Quick Start

```bash
EVAL_MODEL=./experiments/qwen_merged/final_model bash run_eval.sh
```

This runs accuracy (hellaswag, wikitext, RULER at multiple lengths) and latency benchmarks.

## Accuracy

Uses `lm_eval_saga.py`. Examples:

```bash
# hellaswag
python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$EVAL_MODEL \
    --tasks hellaswag --batch_size 4

# RULER (multi-GPU, specific sequence length)
META='{"max_seq_lengths": [4096]}'
accelerate launch --num_processes 8 ./lm_eval_saga.py \
    --model hf --model_args pretrained=$EVAL_MODEL \
    --tasks ruler --batch_size 1 --limit 500 --metadata "$META"

# RULER with atopn (Qwen: --exempt_layers 0,14; Llama: --exempt_layers 0)
accelerate launch --num_processes 8 ./lm_eval_saga.py \
    --model hf --model_args pretrained=$EVAL_MODEL \
    --atopn 512 --exempt_layers 0,14 \
    --tasks ruler --batch_size 1 --limit 500 --metadata "$META"
```

## Latency

Uses `speedup.py` with CUDA graphs and `torch.compile`. Single GPU. Latency depends only on tensor shapes, so the SAGA
student is benchmarked by loading the teacher and merging its key heads at load time (`--merge_mode merge --new_heads N`,
with N = 1 for Qwen and 4 for Llama); the GQA baseline is the unmodified teacher (`--merge_mode baseline`).

```bash
MODEL_DIR=./models/Qwen2.5-1.5B

# SAGA, full attention
python ./speedup.py \
    --model_path $MODEL_DIR --merge_mode merge --new_heads 1 --attention full \
    --batch_sizes 1,4,16,32,64 --seq_lengths 4096,8192,16384,32768,65536,131072 \
    --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
    --cuda_graph --compile_mode default

# SAGA + Atop-N
python ./speedup.py \
    --model_path $MODEL_DIR --merge_mode merge --new_heads 1 --attention atopn:512 \
    --exempt_layers 0,14 \
    --batch_sizes 1,4,16,32,64 --seq_lengths 4096,8192,16384,32768,65536,131072 \
    --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
    --cuda_graph --compile_mode default

# GQA baseline (teacher, original head count)
python ./speedup.py \
    --model_path $MODEL_DIR --merge_mode baseline --attention full \
    --batch_sizes 1,4,16,32,64 --seq_lengths 4096,8192,16384,32768,65536,131072 \
    --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
    --cuda_graph --compile_mode default
```

`speedup.py` exits with a nonzero status if any configuration fails (OOM is reported but not counted as a failure).

## Notes

- For 131072 sequence length, patch `torch._inductor.runtime.hints.TRITON_MAX_BLOCK["X"] = 16384`
- Exempt layers use full attention to avoid NaN from threshold approximation overflow
