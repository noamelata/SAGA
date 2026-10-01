import math
import os
import argparse
import glob
import time
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk, Dataset
import torch._dynamo

from attention.attention import register_sdpa

torch._dynamo.config.optimize_ddp = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

register_sdpa()


def _format_resume_skip_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "unknown"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _ResumeSkipProgress:
    def __init__(self, total_batches: int):
        self.total_batches = total_batches
        self.enabled = total_batches > 0 and int(os.environ.get("RANK", "0")) == 0
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        if self.enabled:
            print(f"Resume data skip: 0/{total_batches} batches (0.0%)", flush=True)

    def update(self, skipped_batches: int):
        if not self.enabled:
            return

        now = time.monotonic()
        is_complete = skipped_batches >= self.total_batches
        if not is_complete and (now - self.last_log_time) < 30:
            return

        elapsed = max(now - self.start_time, 1e-9)
        rate = skipped_batches / elapsed
        percent = 100.0 * skipped_batches / self.total_batches

        if is_complete:
            print(
                f"Resume data skip complete: {skipped_batches}/{self.total_batches} batches "
                f"({percent:.1f}%) in {_format_resume_skip_eta(elapsed)} at {rate:.2f} batches/s",
                flush=True,
            )
        else:
            remaining = max(self.total_batches - skipped_batches, 0)
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"Resume data skip: {skipped_batches}/{self.total_batches} batches "
                f"({percent:.1f}%), {rate:.2f} batches/s, ETA {_format_resume_skip_eta(eta)}",
                flush=True,
            )
        self.last_log_time = now


def _enable_resume_skip_progress_logging():
    try:
        import accelerate.data_loader as accelerate_data_loader
    except ImportError:
        return

    if getattr(accelerate_data_loader, "_resume_skip_logging_patched", False):
        return

    def patched_dataloader_shard_iter(self):
        if self.rng_types is not None:
            accelerate_data_loader.synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()

        self.set_epoch(self.iteration)
        dataloader_iter = self.base_dataloader.__iter__()
        progress = _ResumeSkipProgress(self.skip_batches)
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            self.end()
            return

        batch_index = 0
        while True:
            try:
                if self.device is not None:
                    current_batch = accelerate_data_loader.send_to_device(
                        current_batch, self.device, non_blocking=self._non_blocking
                    )
                self._update_state_dict()
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                else:
                    progress.update(batch_index + 1)
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                self._update_state_dict()
                if batch_index >= self.skip_batches:
                    yield current_batch
                else:
                    progress.update(batch_index + 1)
                break

        self.iteration += 1
        self.end()

    def patched_dataloader_dispatcher_iter(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = None
        if accelerate_data_loader.is_torch_version(">=", "2.0.1"):
            main_iterator = self.base_dataloader.__iter__()
        elif self.state.process_index == 0:
            main_iterator = self.base_dataloader.__iter__()

        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        progress = _ResumeSkipProgress(self.skip_batches)

        while not stop_iteration:
            batch, batch_info = next_batch, next_batch_info

            if self.state.process_index != 0:
                batch = accelerate_data_loader.initialize_tensors(batch_info[0])
            batch = accelerate_data_loader.send_to_device(batch, self.state.device, non_blocking=self._non_blocking)
            batch = accelerate_data_loader.broadcast(batch, from_process=0)

            if not self._drop_last and first_batch is None:
                first_batch = self.slice_fn(
                    batch,
                    slice(0, self.state.num_processes),
                    process_index=self.state.process_index,
                    num_processes=self.state.num_processes,
                )

            if batch is None:
                raise ValueError(
                    f"Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration."
                )

            observed_batch_size = accelerate_data_loader.find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes

            stop_iteration = self._stop_iteration
            if not stop_iteration:
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True

            if not self._drop_last and stop_iteration and observed_batch_size % self.state.num_processes != 0:
                batch = accelerate_data_loader.concatenate([batch, first_batch], dim=0)
                batch_size += 1

            data_slice = slice(self.state.process_index * batch_size, (self.state.process_index + 1) * batch_size)
            batch = self.slice_fn(
                batch,
                data_slice,
                process_index=self.state.process_index,
                num_processes=self.state.num_processes,
            )

            if stop_iteration:
                self.end_of_dataloader = True
                self._update_state_dict()
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            else:
                progress.update(batch_index + 1)
            batch_index += 1

        self.iteration += 1
        self.end()

    def patched_skip_dataloader_iter(self):
        self.begin()
        progress = _ResumeSkipProgress(self.skip_batches)
        for index, batch in enumerate(self.base_dataloader.__iter__()):
            if index >= self.skip_batches:
                self._update_state_dict()
                yield batch
            else:
                progress.update(index + 1)
        self.end()

    accelerate_data_loader.DataLoaderShard.__iter__ = patched_dataloader_shard_iter
    accelerate_data_loader.DataLoaderDispatcher.__iter__ = patched_dataloader_dispatcher_iter
    accelerate_data_loader.SkipDataLoader.__iter__ = patched_skip_dataloader_iter
    accelerate_data_loader._resume_skip_logging_patched = True

def init_weights(module):
    """Default LLM weight initialization (similar to GPT-2/LLaMA)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _ceil_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _closest_divisor(value, target):
    divisors = [d for d in range(1, value + 1) if value % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), -d))


def configure_scratch_attention_layout(
    config,
    num_attention_heads=None,
    num_key_heads=None,
    num_value_heads=None,
    local_rank=0,
):
    base_hidden_size = config.hidden_size
    base_intermediate_size = config.intermediate_size
    base_num_attention_heads = config.num_attention_heads
    base_num_kv_heads = config.num_key_value_heads

    if base_hidden_size % base_num_attention_heads != 0:
        raise ValueError(
            f"Base hidden_size ({base_hidden_size}) must be divisible by "
            f"base num_attention_heads ({base_num_attention_heads})."
        )

    head_dim = base_hidden_size // base_num_attention_heads
    resolved_num_attention_heads = num_attention_heads or base_num_attention_heads
    if resolved_num_attention_heads <= 0:
        raise ValueError("--num-attention-heads must be positive")

    config.num_attention_heads = resolved_num_attention_heads
    config.hidden_size = resolved_num_attention_heads * head_dim

    scaled_intermediate = math.ceil(base_intermediate_size * config.hidden_size / base_hidden_size)
    config.intermediate_size = _ceil_to_multiple(scaled_intermediate, 256)

    scaled_kv_target = round(resolved_num_attention_heads * base_num_kv_heads / base_num_attention_heads)
    default_kv_heads = _closest_divisor(resolved_num_attention_heads, max(1, scaled_kv_target))

    resolved_num_key_heads = num_key_heads or default_kv_heads
    resolved_num_value_heads = num_value_heads or default_kv_heads

    for label, count in (("key", resolved_num_key_heads), ("value", resolved_num_value_heads)):
        if count <= 0:
            raise ValueError(f"--num-{label}-heads must be positive")
        if resolved_num_attention_heads % count != 0:
            raise ValueError(
                f"num_attention_heads ({resolved_num_attention_heads}) must be divisible by "
                f"num_{label}_heads ({count})."
            )

    init_num_kv_heads = math.lcm(resolved_num_key_heads, resolved_num_value_heads)
    if resolved_num_attention_heads % init_num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({resolved_num_attention_heads}) must be divisible by the "
            f"LCM of key/value heads ({init_num_kv_heads})."
        )

    config.num_key_value_heads = init_num_kv_heads
    config.num_key_heads = resolved_num_key_heads
    config.num_value_heads = resolved_num_value_heads

    if local_rank == 0:
        print("\nDerived scratch architecture:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Head dim: {head_dim}")
        print(f"  Attention heads (Q): {resolved_num_attention_heads}")
        print(f"  Init KV heads: {init_num_kv_heads}")
        print(f"  Target key heads: {resolved_num_key_heads}")
        print(f"  Target value heads: {resolved_num_value_heads}")

    return resolved_num_attention_heads, resolved_num_key_heads, resolved_num_value_heads, init_num_kv_heads


def _validate_kv_head_counts(orig_kv_heads, num_key_heads, num_value_heads):
    if num_key_heads > orig_kv_heads or num_value_heads > orig_kv_heads:
        raise ValueError(
            f"Requested key/value heads ({num_key_heads}, {num_value_heads}) exceed "
            f"the original KV head count ({orig_kv_heads})."
        )
    if orig_kv_heads % num_key_heads != 0:
        raise ValueError(
            f"Original KV heads ({orig_kv_heads}) must be divisible by num_key_heads ({num_key_heads})."
        )
    if orig_kv_heads % num_value_heads != 0:
        raise ValueError(
            f"Original KV heads ({orig_kv_heads}) must be divisible by num_value_heads ({num_value_heads})."
        )


def _replace_projection(module, new_out_features):
    has_bias = module.bias is not None
    new_module = nn.Linear(
        module.in_features,
        new_out_features,
        bias=has_bias,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    return new_module


def configure_kv_projection_heads(model, num_key_heads, num_value_heads, local_rank=0):
    """Resize K/V projection modules to match a checkpoint's saved head layout."""
    config = model.config
    orig_kv_heads = config.num_key_value_heads
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    _validate_kv_head_counts(orig_kv_heads, num_key_heads, num_value_heads)

    for layer in model.model.layers:
        k_out_features = num_key_heads * head_dim
        v_out_features = num_value_heads * head_dim

        if layer.self_attn.k_proj.out_features != k_out_features:
            layer.self_attn.k_proj = _replace_projection(layer.self_attn.k_proj, k_out_features)
        if layer.self_attn.v_proj.out_features != v_out_features:
            layer.self_attn.v_proj = _replace_projection(layer.self_attn.v_proj, v_out_features)

    config.num_key_heads = num_key_heads
    config.num_value_heads = num_value_heads

    if local_rank == 0:
        print(f"Configured checkpoint K/V layout: {num_key_heads} key heads, {num_value_heads} value heads")

    return model


def merge_kv_projection_heads(model, num_key_heads, num_value_heads, merge_method="average", local_rank=0):
    """Merge key and value projection heads to specified counts."""
    config = model.config
    orig_kv_heads = config.num_key_value_heads
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    _validate_kv_head_counts(orig_kv_heads, num_key_heads, num_value_heads)

    if local_rank == 0:
        print(f"\nMerging KV heads:")
        print(f"  Original KV heads: {orig_kv_heads}")
        print(f"  New key heads: {num_key_heads} (ratio: {orig_kv_heads}/{num_key_heads} = {orig_kv_heads/num_key_heads:.1f}x)")
        print(f"  New value heads: {num_value_heads} (ratio: {orig_kv_heads}/{num_value_heads} = {orig_kv_heads/num_value_heads:.1f}x)")
        print(f"  Merge method: {merge_method}")

    for layer in model.model.layers:
        # Merge key projection
        k_proj = layer.self_attn.k_proj
        k_has_bias = k_proj.bias is not None
        if merge_method == "average":
            new_k = k_proj.weight.reshape(num_key_heads, -1, head_dim, hidden_size).mean(1)
        elif merge_method == "sum":
            new_k = k_proj.weight.reshape(num_key_heads, -1, head_dim, hidden_size).sum(1)
        else:
            raise ValueError(f"merge_method must be 'average' or 'sum', got {merge_method}")
        new_k_proj = _replace_projection(k_proj, num_key_heads * head_dim)
        new_k_proj.weight = nn.Parameter(new_k.reshape(num_key_heads * head_dim, hidden_size).to(k_proj.weight.dtype))
        if k_has_bias:
            if merge_method == "average":
                new_k_bias = k_proj.bias.reshape(num_key_heads, -1, head_dim).mean(1)
            else:
                new_k_bias = k_proj.bias.reshape(num_key_heads, -1, head_dim).sum(1)
            new_k_proj.bias = nn.Parameter(new_k_bias.reshape(num_key_heads * head_dim).to(k_proj.bias.dtype))
        layer.self_attn.k_proj = new_k_proj

        # Merge value projection
        v_proj = layer.self_attn.v_proj
        v_has_bias = v_proj.bias is not None
        if merge_method == "average":
            new_v = v_proj.weight.reshape(num_value_heads, -1, head_dim, hidden_size).mean(1)
        elif merge_method == "sum":
            new_v = v_proj.weight.reshape(num_value_heads, -1, head_dim, hidden_size).sum(1)
        else:
            raise ValueError(f"merge_method must be 'average' or 'sum', got {merge_method}")
        new_v_proj = _replace_projection(v_proj, num_value_heads * head_dim)
        new_v_proj.weight = nn.Parameter(new_v.reshape(num_value_heads * head_dim, hidden_size).to(v_proj.weight.dtype))
        if v_has_bias:
            if merge_method == "average":
                new_v_bias = v_proj.bias.reshape(num_value_heads, -1, head_dim).mean(1)
            else:
                new_v_bias = v_proj.bias.reshape(num_value_heads, -1, head_dim).sum(1)
            new_v_proj.bias = nn.Parameter(new_v_bias.reshape(num_value_heads * head_dim).to(v_proj.bias.dtype))
        layer.self_attn.v_proj = new_v_proj

    config.num_key_heads = num_key_heads
    config.num_value_heads = num_value_heads
    config.kv_merge_method = merge_method

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SmolLM2-derived model from scratch")

    # Model architecture arguments
    parser.add_argument("--num-attention-heads", type=int, default=None,
                        help="Number of query/attention heads (default: model config)")
    parser.add_argument("--num-key-heads", type=int, default=None,
                        help="Number of key heads (default: use original)")
    parser.add_argument("--num-value-heads", type=int, default=None,
                        help="Number of value heads (default: use original)")
    parser.add_argument("--merge-method", type=str, default="average", choices=["average", "sum"],
                        help="Method for merging KV heads (default: average)")

    # Batch size arguments
    parser.add_argument("--global-batch-size", type=int, default=256,
                        help="Global batch size across all GPUs (default: 128)")
    parser.add_argument("--per-device-batch-size", type=int, default=32,
                        help="Batch size per GPU (default: 2)")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16,
                        help="Eval batch size per GPU (default: 4)")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay (default: 0.1)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum training steps (default: 10000)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps (default: 500)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum sequence length (default: 4096)")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Evaluate and save every N steps (default: 500)")
    parser.add_argument("--dataset-dir", type=str, default="slim",
                        help="Dataset root directory (default: slim)")
    parser.add_argument("--dataset-name", type=str, default="AlppAI/SlimPajama-chunked",
                        help="Hugging Face dataset to stream if no compatible local dataset is found")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Seed used for dataset shuffling and train/validation holdout (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints and final model (default: timestamped experiments dir)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Checkpoint path to resume from, or 'last' to resume the latest checkpoint in output-dir")

    return parser.parse_args()


def preprocess_logits_for_metrics(logits, labels):
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[..., :-1]
    labels = labels[..., 1:]
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / max(mask.sum(), 1)
    return {"accuracy": float(accuracy)}


def main():
    args = parse_args()
    _enable_resume_skip_progress_logging()
    model_name = "HuggingFaceTB/SmolLM2-360M"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = max(1, args.global_batch_size // (args.per_device_batch_size * world_size))

    if local_rank == 0:
        print(f"Configuration:")
        print(f"  Model: {model_name}")
        print(f"  World size: {world_size}")
        print(f"  Global batch size: {args.global_batch_size}")
        print(f"  Per-device batch size: {args.per_device_batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Eval/save steps: {args.eval_steps}")
        print(f"  Dataset dir: {args.dataset_dir}")
        print(f"  Dataset name fallback: {args.dataset_name}")
        print(f"  Data seed: {args.data_seed}")

    # Build experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_parts = ["smollm2-360M", "scratch"]
    if args.num_attention_heads:
        exp_parts.append(f"q{args.num_attention_heads}")
    if args.num_key_heads:
        exp_parts.append(f"k{args.num_key_heads}")
    if args.num_value_heads:
        exp_parts.append(f"v{args.num_value_heads}")
    exp_parts.extend([f"lr{args.learning_rate}", f"bs{args.global_batch_size}"])
    exp_name = "-".join(exp_parts)

    output_dir = args.output_dir or f"./experiments/{timestamp}_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    resume_checkpoint = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "last":
            resume_checkpoint = get_last_checkpoint(output_dir)
            if resume_checkpoint is None:
                raise ValueError(f"No checkpoint found under output-dir: {output_dir}")
        else:
            resume_checkpoint = args.resume_from_checkpoint
        if local_rank == 0:
            print(f"  Resuming from checkpoint: {resume_checkpoint}")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e30)

    # Load config and create model from scratch with random weights
    if local_rank == 0:
        print(f"\nInitializing {model_name} from scratch with random weights...")

    config_source = resume_checkpoint or model_name
    config = AutoConfig.from_pretrained(config_source)

    if resume_checkpoint is not None:
        resume_attention_heads = config.num_attention_heads
        resume_key_heads = getattr(config, "num_key_heads", config.num_key_value_heads)
        resume_value_heads = getattr(config, "num_value_heads", config.num_key_value_heads)

        if args.num_attention_heads is not None and args.num_attention_heads != resume_attention_heads:
            raise ValueError(
                f"--num-attention-heads={args.num_attention_heads} does not match checkpoint "
                f"num_attention_heads={resume_attention_heads}"
            )
        if args.num_key_heads is not None and args.num_key_heads != resume_key_heads:
            raise ValueError(
                f"--num-key-heads={args.num_key_heads} does not match checkpoint num_key_heads={resume_key_heads}"
            )
        if args.num_value_heads is not None and args.num_value_heads != resume_value_heads:
            raise ValueError(
                f"--num-value-heads={args.num_value_heads} does not match checkpoint num_value_heads={resume_value_heads}"
            )

        num_attention_heads = resume_attention_heads
        num_key_heads = resume_key_heads
        num_value_heads = resume_value_heads
    else:
        num_attention_heads, num_key_heads, num_value_heads, init_num_kv_heads = configure_scratch_attention_layout(
            config,
            num_attention_heads=args.num_attention_heads,
            num_key_heads=args.num_key_heads,
            num_value_heads=args.num_value_heads,
            local_rank=local_rank,
        )

    torch.cuda.set_device(local_rank)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.apply(init_weights)

    if local_rank == 0:
        print(f"  Model num_attention_heads: {config.num_attention_heads}")
        print(f"  Model num_key_value_heads: {config.num_key_value_heads}")

    if resume_checkpoint is not None:
        if resume_key_heads != config.num_key_value_heads or resume_value_heads != config.num_key_value_heads:
            model = configure_kv_projection_heads(
                model, resume_key_heads, resume_value_heads, local_rank=local_rank
            )
    elif num_key_heads != init_num_kv_heads or num_value_heads != init_num_kv_heads:
        model = merge_kv_projection_heads(
            model, num_key_heads, num_value_heads,
            merge_method=args.merge_method, local_rank=local_rank
        )

    # Load/prepare dataset

    # -----------------------------------------------------------------------
    # OPTIMIZED DATA LOADING
    # -----------------------------------------------------------------------

    # 1. SHARED FUNCTIONS
    block_size = args.max_length

    def tokenize_raw(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    def pack_examples(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    chunk_path = args.dataset_dir
    slim_train_glob = os.path.join(chunk_path, "train/chunk1", "*.jsonl.zst")
    slim_val_glob = os.path.join(chunk_path, "validation/chunk1", "*.jsonl.zst")
    alppai_glob = os.path.join(chunk_path, "chunk*", "*.jsonl*")

    has_local_slim = bool(glob.glob(slim_train_glob)) and bool(glob.glob(slim_val_glob))
    has_local_alppai = bool(glob.glob(alppai_glob))

    # 2. FAST VALIDATION (Materialized)
    if local_rank == 0:
        print("Preparing validation set...")

    if has_local_slim:
        if local_rank == 0:
            print(f"Using local SlimPajama layout from {chunk_path}")

        val_raw = load_dataset("json", data_files=slim_val_glob, split="train", streaming=True)  # train is not an error
        cols_val = list(next(iter(val_raw)).keys())

        val_tokenized = val_raw.map(tokenize_raw, batched=True, remove_columns=cols_val)
        val_dataset = val_tokenized.map(pack_examples, batched=True, batch_size=16384)

        val_list = list(val_dataset.take(32768))
        val_dataset = Dataset.from_list(val_list)

        if local_rank == 0:
            print("Preparing training stream...")

        train_raw = load_dataset("json", data_files=slim_train_glob, split="train", streaming=True)
        train_shuffled = train_raw.shuffle(seed=args.data_seed, buffer_size=20_000)
        cols_train = list(next(iter(train_raw)).keys())

        train_tokenized = train_shuffled.map(
            tokenize_raw,
            batched=True,
            remove_columns=cols_train
        )

        final_train_dataset = train_tokenized.map(
            pack_examples,
            batched=True,
            batch_size=16384
        )
    else:
        if has_local_alppai:
            if local_rank == 0:
                print(f"Using local AlppAI chunked layout from {chunk_path}")
            raw_dataset = load_dataset("json", data_files=alppai_glob, split="train", streaming=True)
        else:
            if local_rank == 0:
                print(f"Using remote dataset {args.dataset_name}")
            raw_dataset = load_dataset(args.dataset_name, split="train", streaming=True)

        shuffled_dataset = raw_dataset.shuffle(seed=args.data_seed, buffer_size=20_000)
        cols = list(next(iter(raw_dataset)).keys())
        packed_dataset = shuffled_dataset.map(
            tokenize_raw,
            batched=True,
            remove_columns=cols
        ).map(
            pack_examples,
            batched=True,
            batch_size=16384
        )

        # AlppAI only exposes a train split, so hold out packed sequences for validation.
        val_list = list(packed_dataset.take(32768))
        val_dataset = Dataset.from_list(val_list)

        if local_rank == 0:
            print("Preparing training stream...")

        final_train_dataset = packed_dataset.skip(32768)


    # Training arguments with DeepSpeed ZeRO-3 and compilation
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate":0.25},
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        save_steps=args.eval_steps,
        bf16=True,
        torch_compile=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        logging_first_step=True,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if local_rank == 0:
        print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Final evaluation and save
    if local_rank == 0:
        print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()
    if local_rank == 0:
        print(f"Final metrics: {final_metrics}")

    trainer.save_model(os.path.join(output_dir, "final_model"))
    if local_rank == 0:
        print(f"\nModel saved to: {output_dir}/final_model")


if __name__ == "__main__":
    main()
