#!/bin/bash
# SAGA Data Preparation — run once before training.
# Run from the saga/ directory: bash run_prepare_data.sh
#
# IMPORTANT: gqa_distill.py writes the SP tokenized cache to
# $WORKDIR/data_{max_length}_cache/. If you train multiple models (Qwen + Llama)
# with the same WORKDIR, you MUST rebuild the caches for each model since the
# tokenizer differs. Use separate WORKDIRs per model, or delete the existing
# caches before switching models.
set -eo pipefail

# ===================== USER CONFIG =====================
MODEL_ARCH=qwen          # qwen | llama

# Working directory: where models/, experiments/, and output data caches will live.
# Default is the parent of the saga/ directory. Set to an absolute path if needed.
WORKDIR=${WORKDIR:-.}

if [ "$MODEL_ARCH" = "qwen" ]; then
    MODEL_DIR=$WORKDIR/models/Qwen2.5-1.5B
    NUM_KEY_HEADS=1
elif [ "$MODEL_ARCH" = "llama" ]; then
    MODEL_DIR=$WORKDIR/models/Llama-3.2-1B
    NUM_KEY_HEADS=4
fi
# =======================================================

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}

echo "SAGA data preparation: MODEL_ARCH=$MODEL_ARCH  WORKDIR=$WORKDIR"

# ---- Stage 1: 4K data ----
echo "--- Building 4K SP cache ---"
torchrun --nproc_per_node=1 ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $NUM_KEY_HEADS \
    --alpha 0.7 --learning-rate 2e-4 --temperature 3.0 \
    --max-length 4096 --max-steps 0 \
    --per-device-batch-size 4 \
    --data-cache-dir $WORKDIR/data_4096_cache \
    --model-name $MODEL_DIR \
    --output-dir /tmp/saga_dummy_4k 2>&1 | tail -5
rm -rf /tmp/saga_dummy_4k

echo "--- Generating 4K NIAH (200K examples) ---"
python ./data/generate_niah_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 4096 \
    --num-examples 200000 \
    --output-dir $WORKDIR/data_niah_synthetic_4k \
    --seed 42

echo "--- Building 4K mixed cache (75/15/10) ---"
python ./data/prepare_mixed_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 4096 \
    --output-dir $WORKDIR/data_4096_mixed_75_15_10_cache \
    --slimpajama-cache $WORKDIR/data_4096_cache/tokenized_slimpajama_packed \
    --extra-datasets hotpotqa/hotpot_qa rajpurkar/squad_v2 \
    --extra-ratio 0.15 \
    --niah-dataset $WORKDIR/data_niah_synthetic_4k/tokenized_niah_synthetic \
    --niah-ratio 0.10 \
    --seed 42

# ---- Stage 2: 20K data ----
echo "--- Building 20K SP cache ---"
torchrun --nproc_per_node=1 ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $NUM_KEY_HEADS \
    --alpha 0.7 --learning-rate 2e-4 --temperature 3.0 \
    --max-length 20480 --max-steps 0 \
    --per-device-batch-size 1 \
    --data-cache-dir $WORKDIR/data_20480_cache \
    --model-name $MODEL_DIR \
    --output-dir /tmp/saga_dummy_20k 2>&1 | tail -5
rm -rf /tmp/saga_dummy_20k

echo "--- Generating 20K NIAH (50K examples) ---"
python ./data/generate_niah_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 20480 \
    --num-examples 50000 \
    --output-dir $WORKDIR/data_niah_synthetic_20k \
    --seed 42

echo "--- Building 20K mixed cache (75/5/10/10) ---"
python ./data/prepare_mixed_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 20480 \
    --output-dir $WORKDIR/data_20480_mixed_75_5_10_10_cache \
    --slimpajama-cache $WORKDIR/data_20480_cache/tokenized_slimpajama_packed \
    --extra-datasets hotpotqa/hotpot_qa rajpurkar/squad_v2 \
    --extra-ratio 0.05 \
    --unpacked-datasets \
        THUDM/LongAlign-10k \
        Yukang/LongAlpaca-12k \
    --unpacked-ratio 0.10 \
    --niah-dataset $WORKDIR/data_niah_synthetic_20k/tokenized_niah_synthetic \
    --niah-ratio 0.10 \
    --seed 42

# ---- Stage 3: 36K data ----
echo "--- Building 36K SP cache ---"
torchrun --nproc_per_node=1 ./gqa_distill.py \
    --merge-kv-heads --num-key-heads $NUM_KEY_HEADS \
    --alpha 0.7 --learning-rate 5e-5 --temperature 3.0 \
    --max-length 36864 --max-steps 0 \
    --per-device-batch-size 1 \
    --data-cache-dir $WORKDIR/data_36864_cache \
    --model-name $MODEL_DIR \
    --output-dir /tmp/saga_dummy_36k 2>&1 | tail -5
rm -rf /tmp/saga_dummy_36k

echo "--- Generating 36K NIAH (30K examples) ---"
python ./data/generate_niah_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 36864 \
    --num-examples 30000 \
    --output-dir $WORKDIR/data_niah_synthetic_36k \
    --seed 42

echo "--- Building 36K mixed cache (75/5/10/10) ---"
python ./data/prepare_mixed_data.py \
    --tokenizer $MODEL_DIR \
    --max-length 36864 \
    --output-dir $WORKDIR/data_36864_mixed_75_5_10_10_cache \
    --slimpajama-cache $WORKDIR/data_36864_cache/tokenized_slimpajama_packed \
    --extra-datasets hotpotqa/hotpot_qa rajpurkar/squad_v2 \
    --extra-ratio 0.05 \
    --unpacked-datasets \
        THUDM/LongAlign-10k \
        Yukang/LongAlpaca-12k \
    --unpacked-ratio 0.10 \
    --niah-dataset $WORKDIR/data_niah_synthetic_36k/tokenized_niah_synthetic \
    --niah-ratio 0.10 \
    --seed 42

echo "Data preparation complete."
