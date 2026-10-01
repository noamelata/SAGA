#!/bin/bash
# SAGA Full Evaluation — accuracy + latency benchmarks.
# Run from the saga/ directory:
#   EVAL_MODEL=./experiments/qwen_merged/final_model bash run_eval.sh
set -eo pipefail

# ===================== USER CONFIG =====================
MODEL_ARCH=${MODEL_ARCH:-qwen}          # qwen | llama

# Working directory: where models/, experiments/, and logs/ live.
WORKDIR=${WORKDIR:-.}

EVAL_MODEL=${EVAL_MODEL:-$WORKDIR/experiments/${MODEL_ARCH}_merged/final_model}
LOGS_DIR=${LOGS_DIR:-$WORKDIR/logs}

# Teacher model path. Latency is benchmarked on the teacher, either as-is (GQA baseline)
# or with key heads merged at load time (SAGA student architecture).
if [ "$MODEL_ARCH" = "qwen" ]; then
    MODEL_DIR=$WORKDIR/models/Qwen2.5-1.5B
    NUM_KEY_HEADS=1
elif [ "$MODEL_ARCH" = "llama" ]; then
    MODEL_DIR=$WORKDIR/models/Llama-3.2-1B
    NUM_KEY_HEADS=4
fi
NUM_GPUS=${NUM_GPUS:-8}
# =======================================================

export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p $LOGS_DIR

if [ ! -d "$EVAL_MODEL" ]; then
    echo "ERROR: Model not found at $EVAL_MODEL"
    exit 1
fi

echo "=== SAGA Evaluation: $EVAL_MODEL ==="

# Qwen-specific: exempt layers 0 and 14 from atopn approximation
if [ "$MODEL_ARCH" = "qwen" ]; then
    EXEMPT_ARGS="--exempt_layers 0,14"
elif [ "$MODEL_ARCH" = "llama" ]; then
    EXEMPT_ARGS="--exempt_layers 0"
fi

# ---- Accuracy: hellaswag ----
echo "--- hellaswag ---"
CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$EVAL_MODEL \
    --tasks hellaswag --batch_size 4 \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merged_hellaswag.log

# ---- Accuracy: wikitext ----
echo "--- wikitext ---"
CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$EVAL_MODEL \
    --tasks wikitext --batch_size 1 \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merged_wikitext.log

# ---- Accuracy: RULER full attention (500 samples) ----
for SEQ_LEN in 8192 16384 32768; do
    LEN_TAG=$(( SEQ_LEN / 1024 ))K
    echo "--- RULER full ${LEN_TAG} (500 samples) ---"
    META="{\"max_seq_lengths\": [$SEQ_LEN]}"
    accelerate launch --num_processes $NUM_GPUS ./lm_eval_saga.py \
        --model hf --model_args pretrained=$EVAL_MODEL \
        --tasks ruler --batch_size 1 --limit 500 --metadata "$META" \
        2>&1 | tee "$LOGS_DIR/${MODEL_ARCH}_merged_ruler_full_${LEN_TAG}.log"
done

# ---- Accuracy: RULER atopn (500 samples) ----
for ATOPN in 512 1024 2048; do
    for SEQ_LEN in 8192 16384 32768; do
        LEN_TAG=$(( SEQ_LEN / 1024 ))K
        echo "--- RULER atopn:${ATOPN} ${LEN_TAG} (500 samples) ---"
        META="{\"max_seq_lengths\": [$SEQ_LEN]}"
        accelerate launch --num_processes $NUM_GPUS ./lm_eval_saga.py \
            --model hf --model_args pretrained=$EVAL_MODEL \
            --atopn $ATOPN $EXEMPT_ARGS \
            --tasks ruler --batch_size 1 --limit 500 --metadata "$META" \
            2>&1 | tee "$LOGS_DIR/${MODEL_ARCH}_merged_ruler_atopn${ATOPN}_${LEN_TAG}.log"
    done
done

# ---- Latency: SAGA (teacher with key heads merged at load time; latency depends only on shapes) ----
LATENCY_SEQ=4096,8192,16384,32768,65536,131072
LATENCY_BS=1,4,16

echo "--- Latency: merged student, full attention ---"
CUDA_VISIBLE_DEVICES=0 python ./speedup.py \
    --model_path $MODEL_DIR \
    --merge_mode merge --new_heads $NUM_KEY_HEADS \
    --attention full \
    --batch_sizes $LATENCY_BS \
    --seq_lengths $LATENCY_SEQ \
    --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
    --cuda_graph --compile_mode default \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merged_latency_merged_full.log

# ---- Latency: merged student, atopn ----
for ATOPN in 512 1024 2048; do
    echo "--- Latency: merged student, atopn:${ATOPN} ---"
    CUDA_VISIBLE_DEVICES=0 python ./speedup.py \
        --model_path $MODEL_DIR \
        --merge_mode merge --new_heads $NUM_KEY_HEADS \
        --attention atopn:$ATOPN \
        $EXEMPT_ARGS \
        --batch_sizes $LATENCY_BS \
        --seq_lengths $LATENCY_SEQ \
        --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
        --cuda_graph --compile_mode default \
        2>&1 | tee "$LOGS_DIR/${MODEL_ARCH}_merged_latency_merged_atopn${ATOPN}.log"
done

# ---- Latency: baseline teacher, full attention ----
echo "--- Latency: baseline teacher, full attention ---"
CUDA_VISIBLE_DEVICES=0 python ./speedup.py \
    --model_path $MODEL_DIR \
    --merge_mode baseline \
    --attention full \
    --batch_sizes $LATENCY_BS \
    --seq_lengths $LATENCY_SEQ \
    --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
    --cuda_graph --compile_mode default \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merged_latency_baseline_full.log

# ---- Latency: baseline teacher, atopn ----
for ATOPN in 512 1024 2048; do
    echo "--- Latency: baseline teacher, atopn:${ATOPN} ---"
    CUDA_VISIBLE_DEVICES=0 python ./speedup.py \
        --model_path $MODEL_DIR \
        --merge_mode baseline \
        --attention atopn:$ATOPN \
        $EXEMPT_ARGS \
        --batch_sizes $LATENCY_BS \
        --seq_lengths $LATENCY_SEQ \
        --num_warmup 5 --num_iterations 100 --num_new_tokens 8 \
        --cuda_graph --compile_mode default \
        2>&1 | tee "$LOGS_DIR/${MODEL_ARCH}_merged_latency_baseline_atopn${ATOPN}.log"
done

echo "=== Evaluation complete. Logs in $LOGS_DIR/${MODEL_ARCH}_merged_* ==="
