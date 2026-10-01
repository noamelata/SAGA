#!/bin/bash
# SAGA from-scratch training (Section 5.2): SmolLM2-360M with asymmetric KV heads.
# Run from the saga/ directory: bash train_scratch.sh
set -eo pipefail

OUTPUT_DIR="./experiments/smollm2_16_1_8_10k/"
mkdir -p "${OUTPUT_DIR}"

NUM_GPUS=${NUM_GPUS:-4}

export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}

torchrun --master_port=$((29500 + RANDOM % 1000)) \
  --nproc_per_node=$NUM_GPUS --nnodes=1 ./train_scratch.py \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-dir ${DATASET_DIR:-./data/SlimPajama-chunked} \
  --global-batch-size 256 \
  --per-device-batch-size 64 \
  --per-device-eval-batch-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --warmup-steps 500 \
  --max-steps 20000 \
  --eval-steps 500 \
  --max-length 1024 \
  --num-attention-heads 16 \
  --num-key-heads 1 \
  --num-value-heads 8 2>&1 | tee ${OUTPUT_DIR}/train.log
