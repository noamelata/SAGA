# SAGA Fine-tuning Guide

Distills a pre-trained causal LM into a version with fewer KV heads via knowledge distillation. Supported models: Qwen2.5-1.5B, Llama-3.2-1B.

## Prerequisites

- 8x GPU (40 GB for Stage 1, 80 GB for Stages 2-3)
- Models downloaded from HuggingFace or placed at `$WORKDIR/models/<name>/`
- Datasets downloaded automatically from HuggingFace Hub

## Usage

All scripts run from **inside `saga/`**. Set `MODEL_ARCH` (qwen/llama) at the top of each script. Set `WORKDIR` to override the default (`.`).

```bash
bash run_prepare_data.sh   # build data caches
bash finetune.sh           # student init + stages 1-3
```

## Pipeline

1. **Data preparation** (`run_prepare_data.sh`): builds SP caches, synthetic NIAH data, and mixed datasets for each stage.
2. **Student init** (`merge_heads.py`): data-driven KV head merging via minimum-weight matching.
3. **Stage 1** (4K, 15K steps): distillation at 4096 context. Output: `experiments/${MODEL_ARCH}_stage1_4k/`
4. **Stage 2** (20K, 5K steps): extends context to 20480. Output: `experiments/${MODEL_ARCH}_stage2_20k/`
5. **Stage 3** (36K, 2-4K steps): mixed-length with interleaved 20K+36K batches. Output: `experiments/${MODEL_ARCH}_merged/`

**Note**: SP tokenized caches are model-specific. Use separate `WORKDIR` per model or delete caches when switching.

## Hyperparameters

| Param | Stage 1 | Stage 2 | Stage 3 (Qwen) | Stage 3 (Llama) |
|-------|---------|---------|---------------|----------------|
| context | 4096 | 20480 | 36864 | 36864 |
| steps | 15000 | 5000 | 2000 | 4000 |
| lr | 2e-4 | 1e-4 | 5e-5 | 5e-5 |
| alpha | 0.7 | 0.7 | 0.7 | 0.85 |
| temperature | 3.0 | 3.0 | 3.0 | 4.0 |
| warmup | 200 | 200 | 100 | 200 |
| decay | 5000 | 1500 | 500 | 800 |
| batch (global) | 64 | 32 | 32 | 32 |
