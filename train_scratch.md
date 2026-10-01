# From-Scratch Training (Section 5.2)

Trains a SmolLM2-360M model from scratch with asymmetric KV head counts (fewer keys than values).

## Usage

```bash
bash train_scratch.sh
```

Or with custom settings:

```bash
NUM_GPUS=8 torchrun --nproc_per_node=8 ./train_scratch.py \
    --num-attention-heads 16 --num-key-heads 1 --num-value-heads 8 \
    --max-length 1024 --max-steps 20000 \
    --global-batch-size 256 --per-device-batch-size 32 \
    --output-dir ./experiments/smollm2_16_1_8
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-attention-heads` | model config | Query head count (determines hidden size) |
| `--num-key-heads` | auto | Key heads after merge |
| `--num-value-heads` | auto | Value heads (can differ from key heads) |
| `--max-length` | 1024 | Sequence length |
| `--dataset-name` | `AlppAI/SlimPajama-chunked` | HuggingFace dataset (streamed if no local data) |
| `--dataset-dir` | `./data/SlimPajama-chunked` | Local dataset path (falls back to `--dataset-name`) |

