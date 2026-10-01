#!/bin/bash
# SAGA: Full training pipeline — student init and stages 1, 2, 3 (with quick evals after stages 1 and 2).
# For the final evaluation, run run_eval.sh afterwards.
# Run from the saga/ directory: bash finetune.sh
#
# Prerequisites: run run_prepare_data.sh first to build all data caches.
set -eo pipefail

# ===================== USER CONFIG =====================
MODEL_ARCH=qwen          # qwen | llama

# Working directory: where models/, experiments/, data_*_cache/, logs/ live.
# Default is the parent of the saga/ directory. Set to an absolute path if needed.
WORKDIR=${WORKDIR:-.}

if [ "$MODEL_ARCH" = "qwen" ]; then
    MODEL_DIR=$WORKDIR/models/Qwen2.5-1.5B
    NUM_KEY_HEADS=1
elif [ "$MODEL_ARCH" = "llama" ]; then
    MODEL_DIR=$WORKDIR/models/Llama-3.2-1B
    NUM_KEY_HEADS=4
fi
NUM_GPUS=${NUM_GPUS:-8}
LOGS_DIR=$WORKDIR/logs
# =======================================================

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}

mkdir -p $LOGS_DIR

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    exit 1
fi

# Model-specific Stage 3 hyperparameters
if [ "$MODEL_ARCH" = "qwen" ]; then
    S3_NUM_KEY_HEADS=1; S3_ALPHA=0.7; S3_TEMPERATURE=3.0
    S3_MAX_STEPS=2000; S3_WARMUP=100; S3_DECAY=500
elif [ "$MODEL_ARCH" = "llama" ]; then
    S3_NUM_KEY_HEADS=4; S3_ALPHA=0.85; S3_TEMPERATURE=4.0
    S3_MAX_STEPS=4000; S3_WARMUP=200; S3_DECAY=800
else
    echo "ERROR: Unknown MODEL_ARCH=$MODEL_ARCH (use qwen or llama)"
    exit 1
fi

# ===================== STUDENT INIT: matching-based merge =====================
# Data-driven minimum-weight matching to find optimal head pairs (Sec 4.1).
INIT_DIR=$WORKDIR/experiments/${MODEL_ARCH}_student_init

echo "============================================"
echo "  Student init: merge KV heads -> $INIT_DIR"
echo "============================================"

python ./merge_heads.py \
    --model-path $MODEL_DIR \
    --data-cache-dir $WORKDIR/data_4096_cache/tokenized_slimpajama_packed \
    --output-dir $INIT_DIR \
    --max-length 4096 \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merge_heads.log

# ===================== STAGE 1: 4K, 15K steps =====================
STAGE1_DIR=$WORKDIR/experiments/${MODEL_ARCH}_stage1_4k

echo "============================================"
echo "  Stage 1: 4K, 15K steps, ${NUM_KEY_HEADS} KV head(s)"
echo "  Output: $STAGE1_DIR"
echo "============================================"

torchrun --nproc_per_node=$NUM_GPUS ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $NUM_KEY_HEADS \
    --alpha 0.7 --learning-rate 2e-4 --temperature 3.0 \
    --lr-scheduler-type warmup_stable_decay \
    --warmup-steps 200 \
    --num-decay-steps 5000 \
    --max-length 4096 --max-steps 15000 \
    --per-device-batch-size 4 \
    --global-batch-size 64 \
    --data-cache-dir $WORKDIR/data_4096_mixed_75_15_10_cache \
    --model-name $MODEL_DIR \
    --student-model-name $INIT_DIR \
    --output-dir $STAGE1_DIR \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage1_train.log

CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$STAGE1_DIR/final_model \
    --tasks hellaswag --batch_size 4 2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage1_hellaswag.log

CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$STAGE1_DIR/final_model \
    --tasks wikitext --batch_size 1 2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage1_wikitext.log

META='{"max_seq_lengths": [4096]}'
accelerate launch --num_processes $NUM_GPUS ./lm_eval_saga.py \
    --model hf --model_args pretrained=$STAGE1_DIR/final_model \
    --tasks ruler --batch_size 1 --limit 50 --metadata "$META" \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage1_ruler_4K.log

# ===================== STAGE 2: 20K, 5K steps =====================
STAGE2_DIR=$WORKDIR/experiments/${MODEL_ARCH}_stage2_20k

echo "============================================"
echo "  Stage 2: 20K, 5K steps"
echo "  Student: $STAGE1_DIR/final_model  ->  $STAGE2_DIR"
echo "============================================"

torchrun --nproc_per_node=$NUM_GPUS ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $NUM_KEY_HEADS \
    --alpha 0.7 --learning-rate 1e-4 --temperature 3.0 \
    --lr-scheduler-type warmup_stable_decay \
    --warmup-steps 200 \
    --num-decay-steps 1500 \
    --max-length 20480 --max-steps 5000 \
    --per-device-batch-size 1 \
    --global-batch-size 32 \
    --data-cache-dir $WORKDIR/data_20480_mixed_75_5_10_10_cache \
    --model-name $MODEL_DIR \
    --student-model-name $STAGE1_DIR/final_model \
    --output-dir $STAGE2_DIR \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage2_train.log

CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$STAGE2_DIR/final_model \
    --tasks hellaswag --batch_size 4 2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage2_hellaswag.log

CUDA_VISIBLE_DEVICES=0 python ./lm_eval_saga.py \
    --model hf --model_args pretrained=$STAGE2_DIR/final_model \
    --tasks wikitext --batch_size 1 2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_stage2_wikitext.log

for SEQ_LEN in 4096 8192 16384; do
    LEN_TAG=$(( SEQ_LEN / 1024 ))K
    META="{\"max_seq_lengths\": [$SEQ_LEN]}"
    accelerate launch --num_processes $NUM_GPUS ./lm_eval_saga.py \
        --model hf --model_args pretrained=$STAGE2_DIR/final_model \
        --tasks ruler --batch_size 1 --limit 50 --metadata "$META" \
        2>&1 | tee "$LOGS_DIR/${MODEL_ARCH}_stage2_ruler_${LEN_TAG}.log"
done

# ===================== STAGE 3: 36K, mixed-length =====================
STAGE3_DIR=$WORKDIR/experiments/${MODEL_ARCH}_merged

echo "============================================"
echo "  Stage 3: 36K, $S3_MAX_STEPS steps (${MODEL_ARCH})"
echo "  Student: $STAGE2_DIR/final_model  ->  $STAGE3_DIR"
echo "============================================"

torchrun --nproc_per_node=$NUM_GPUS ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $S3_NUM_KEY_HEADS \
    --alpha $S3_ALPHA --learning-rate 5e-5 --temperature $S3_TEMPERATURE \
    --lr-scheduler-type warmup_stable_decay \
    --warmup-steps $S3_WARMUP \
    --num-decay-steps $S3_DECAY \
    --max-length 36864 --max-steps $S3_MAX_STEPS \
    --per-device-batch-size 1 \
    --global-batch-size 32 \
    --data-cache-dir $WORKDIR/data_36864_mixed_75_5_10_10_cache \
    --aux-data-cache-dir $WORKDIR/data_20480_mixed_75_5_10_10_cache \
    --aux-microbatches 2 \
    --model-name $MODEL_DIR \
    --student-model-name $STAGE2_DIR/final_model \
    --output-dir $STAGE3_DIR \
    2>&1 | tee $LOGS_DIR/${MODEL_ARCH}_merged_train.log


echo "============================================"
echo "  Pipeline complete: $STAGE3_DIR/final_model"
echo "  Evaluate with: EVAL_MODEL=$STAGE3_DIR/final_model MODEL_ARCH=$MODEL_ARCH bash run_eval.sh"
echo "  Logs: $LOGS_DIR/${MODEL_ARCH}_*"
echo "============================================"
