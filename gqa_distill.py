import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk
from transformers import TrainerCallback
import gc
import numpy as np

from attention.attention import register_sdpa


register_sdpa()


class MemoryCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        """Clear cache after each evaluation"""
        torch.cuda.empty_cache()
        gc.collect()


def freeze_non_attention_parameters(model, local_rank=0):
    """
    Freeze all parameters except those in attention layers.
    This includes q_proj, k_proj, v_proj, and o_proj in self_attn modules.
    """
    total_params = 0
    trainable_params = 0

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
        total_params += param.numel()

    # Then unfreeze attention parameters
    for layer in model.model.layers:
        # Unfreeze all parameters in self_attn
        for name, param in layer.self_attn.named_parameters():
            param.requires_grad = True
            trainable_params += param.numel()

    if local_rank == 0:
        print(f"Frozen non-attention parameters")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    return model


def merge_key_projection_heads(model, new_heads=1, local_rank=0):
    """Merge key projection heads by averaging and replacing each k_proj with a smaller nn.Linear."""
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    if new_heads > num_kv_heads:
        raise ValueError(
            f"num_key_heads ({new_heads}) must be <= num_key_value_heads ({num_kv_heads}). "
            f"Cannot merge {num_kv_heads} heads into {new_heads}."
        )
    if num_kv_heads % new_heads != 0:
        raise ValueError(
            f"num_key_value_heads ({num_kv_heads}) must be divisible by num_key_heads ({new_heads})."
        )

    heads_per_group = num_kv_heads // new_heads

    for layer in model.model.layers:
        k_proj = layer.self_attn.k_proj
        has_bias = k_proj.bias is not None

        new_weight = k_proj.weight.reshape(new_heads, heads_per_group, head_dim, hidden_size).mean(1).reshape(
            new_heads * head_dim, hidden_size)

        new_bias_data = None
        if has_bias:
            new_bias_data = k_proj.bias.reshape(new_heads, heads_per_group, head_dim).mean(1).reshape(new_heads * head_dim)

        new_k_proj = nn.Linear(
            hidden_size, new_heads * head_dim, bias=has_bias,
            device=k_proj.weight.device, dtype=k_proj.weight.dtype,
        )
        new_k_proj.weight = nn.Parameter(new_weight.to(k_proj.weight.dtype), requires_grad=True)
        if has_bias and new_bias_data is not None:
            new_k_proj.bias = nn.Parameter(new_bias_data.to(k_proj.bias.dtype), requires_grad=True)

        layer.self_attn.k_proj = new_k_proj

    config.num_key_heads = new_heads

    if local_rank == 0:
        print(f"Merged k heads by averaging ({num_kv_heads} → {new_heads} key heads)")
    return model


def pack_sequences(tokenized_dataset, max_length, eos_token_id):
    """
    Pack tokenized sequences into full-length chunks separated by EOS tokens.
    This avoids wasting compute on padding tokens.
    """
    from tqdm import tqdm

    all_ids = tokenized_dataset["input_ids"]
    n_docs = len(all_ids)

    print(f"    Packing {n_docs} docs into chunks of {max_length}...")
    pieces = []
    for i, ids in tqdm(enumerate(all_ids), total=n_docs, desc="    Packing"):
        if i > 0:
            pieces.append(eos_token_id)
        pieces.extend(ids)

    all_tokens = np.array(pieces, dtype=np.int32)
    del pieces

    num_chunks = len(all_tokens) // max_length
    all_tokens = all_tokens[:num_chunks * max_length]
    chunked = all_tokens.reshape(num_chunks, max_length)

    print(f"    Created {num_chunks} chunks of {max_length} tokens")
    from datasets import Dataset
    return Dataset.from_dict({
        "input_ids": chunked.tolist(),
        "attention_mask": [[1] * max_length] * num_chunks,
        "labels": chunked.tolist(),
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune llama model with knowledge distillation")

    # Model modification arguments
    parser.add_argument("--merge-kv-heads", action="store_true",
                        help="Whether to merge key projection heads")
    parser.add_argument("--num-key-heads", type=int, default=1,
                        help="Number of key heads after merging (default: %(default)s)")
    parser.add_argument("--train-only-attention", action="store_true",
                        help="Only train attention layers, freeze all other parameters")

    # Knowledge distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for knowledge distillation (default: %(default)s)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss vs hard label loss (default: %(default)s)")
    # Batch size arguments
    parser.add_argument("--global-batch-size", type=int, default=128,
                        help="Global batch size across all GPUs (default: %(default)s)")
    parser.add_argument("--per-device-batch-size", type=int, default=2,
                        help="Maximum batch size per GPU (default: %(default)s)")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1,
                        help="Batch size per GPU for evaluation (default: %(default)s)")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum number of training steps (default: %(default)s)")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Number of warmup steps (default: %(default)s)")

    # Data arguments
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Maximum sequence length (default: %(default)s)")
    parser.add_argument("--data-cache-dir", type=str, default=None,
                        help="Path to pre-built data cache. Overrides default SlimPajama cache. "
                             "Must contain tokenized_slimpajama_packed/ and tokenized_slimpajama_val_packed/")
    parser.add_argument("--lr-scheduler-type", type=str, default=None,
                        help="Override LR scheduler (e.g. cosine, warmup_stable_decay). Default: cosine")
    parser.add_argument("--num-decay-steps", type=int, default=None,
                        help="Number of decay steps for warmup_stable_decay scheduler")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (faster but uses more memory)")

    # Other arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Model name or path (default: %(default)s)")
    parser.add_argument("--student-model-name", type=str, default=None,
                        help="Student model name or path (default: same as --model-name)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints and logs (default: timestamped experiments dir)")
    parser.add_argument("--resume-from-checkpoint", action="store_true",
                        help="Resume from the latest checkpoint in output_dir")

    # Mixed-length training arguments
    parser.add_argument("--aux-data-cache-dir", type=str, default=None,
                        help="Path to auxiliary data cache for mixed-length training. "
                             "Microbatches alternate between main and aux data within each gradient step.")
    parser.add_argument("--aux-microbatches", type=int, default=2,
                        help="Number of microbatches per gradient step from aux data (default: %(default)s)")

    args = parser.parse_args()
    if args.student_model_name is None:
        args.student_model_name = args.model_name
    return args


def _compute_ce_chunk(student_chunk, labels_chunk, mask_chunk):
    """Compute CE loss for a single sequence chunk (used with gradient checkpointing)."""
    vocab_size = student_chunk.size(-1)
    logits = student_chunk.float().reshape(-1, vocab_size)
    labels = labels_chunk.reshape(-1)
    m = mask_chunk.reshape(-1)
    per_token = F.cross_entropy(logits, labels, reduction='none', ignore_index=-100)
    return (per_token * m).sum()


def compute_ce_loss(student_logits, shift_labels, mask, chunk_size=1024):
    """Compute cross-entropy loss with proper masking, chunked to save memory."""
    from torch.utils.checkpoint import checkpoint

    num_valid = mask.sum().clamp(min=1.0)
    total_loss = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)
    seq_len = student_logits.size(1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_loss = checkpoint(
            _compute_ce_chunk,
            student_logits[:, start:end, :],
            shift_labels[:, start:end],
            mask[:, start:end],
            use_reentrant=False,
        )
        total_loss = total_loss + chunk_loss

    return total_loss / num_valid


def _compute_kl_chunk(student_chunk, teacher_chunk, mask_chunk, temperature):
    """Compute KL divergence for a single sequence chunk (used with gradient checkpointing)."""
    vocab_size = student_chunk.size(-1)
    s = student_chunk.float().reshape(-1, vocab_size) / temperature
    t = teacher_chunk.float().reshape(-1, vocab_size) / temperature
    s_log_probs = F.log_softmax(s, dim=-1)
    t_probs = F.softmax(t, dim=-1)
    kl_per_token = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1)
    return (kl_per_token * mask_chunk.reshape(-1)).sum()


def compute_kl_loss(student_logits, teacher_logits, mask, temperature, chunk_size=1024):
    """Compute KL divergence loss with proper masking, chunked to save memory."""
    from torch.utils.checkpoint import checkpoint

    num_valid = mask.sum().clamp(min=1.0)
    total_kl = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)
    seq_len = student_logits.size(1)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_kl = checkpoint(
            _compute_kl_chunk,
            student_logits[:, start:end, :],
            teacher_logits[:, start:end, :],
            mask[:, start:end],
            temperature,
            use_reentrant=False,
        )
        total_kl = total_kl + chunk_kl

    return (total_kl / num_valid) * (temperature ** 2)


class _InterleavedIterableDataset(torch.utils.data.IterableDataset):
    """Wraps InterleavedDataLoader as an IterableDataset so skip_first_batches uses the
    simple SkipDataLoader path (skips N items from the iterable) rather than trying to
    rebuild a sampler-based DataLoader from the underlying map-style dataset."""
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        return iter(self._loader)


class InterleavedDataLoader:
    """Yields microbatches alternating between two dataloaders in a fixed pattern.

    Pattern per gradient step: [aux]*aux_count + [main]*main_count.
    This ensures each gradient update includes data from both lengths.
    """
    def __init__(self, dl_main, dl_aux, main_count=2, aux_count=2):
        self.dl_main = dl_main
        self.dl_aux = dl_aux
        self.cycle = aux_count + main_count
        self.aux_count = aux_count
        self._iterable_ds = _InterleavedIterableDataset(self)

    def __iter__(self):
        it_main = iter(self.dl_main)
        it_aux = iter(self.dl_aux)
        step = 0
        while True:
            try:
                if step % self.cycle < self.aux_count:
                    yield next(it_aux)
                else:
                    yield next(it_main)
                step += 1
            except StopIteration:
                return

    def __len__(self):
        n_main = len(self.dl_main)
        n_aux = len(self.dl_aux)
        main_count = self.cycle - self.aux_count
        return min(
            n_main * self.cycle // main_count,
            n_aux * self.cycle // self.aux_count
        )

    @property
    def dataset(self):
        return self._iterable_ds

    @property
    def batch_size(self):
        # Tells skip_first_batches to request 1 item at a time; our iterable
        # already yields complete batches so collate_fn unwraps the singleton list.
        return 1

    @property
    def collate_fn(self):
        return lambda x: x[0]

    @property
    def drop_last(self):
        return getattr(self.dl_main, 'drop_last', False)

    @property
    def num_workers(self):
        # Must be 0: SkipDataLoader would otherwise try to spawn subprocesses from
        # within existing DataLoader daemon workers, which is not allowed.
        # The underlying dl_main/dl_aux already handle their own workers.
        return 0

    @property
    def prefetch_factor(self):
        return None  # Only valid when num_workers > 0

    @property
    def pin_memory(self):
        # Tensors yielded by dl_main/dl_aux are already on the correct device;
        # SkipDataLoader must not try to re-pin them.
        return False

    @property
    def persistent_workers(self):
        return getattr(self.dl_main, 'persistent_workers', False)

    @property
    def worker_init_fn(self):
        return getattr(self.dl_main, 'worker_init_fn', None)

    @property
    def timeout(self):
        return getattr(self.dl_main, 'timeout', 0)


class DistillationTrainer(Trainer):
    def __init__(self, model, args, *o_args, teacher_model=None, temperature=2.0, alpha=0.5,
                 max_steps=None, aux_train_dataset=None, aux_microbatches=2, **kwargs):
        super().__init__(model, args, *o_args, **kwargs)
        self._move_model_to_device(teacher_model, args.device)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.aux_train_dataset = aux_train_dataset
        self.aux_microbatches = aux_microbatches

        # Move teacher to same device as student and set to eval mode
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Loss accumulation for correct logging with gradient accumulation
        self._loss_accum = {"hard_loss": 0.0, "kl_loss": 0.0}
        self._micro_steps = 0

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if optimizer is None:
            return None
        non_empty_groups = [group for group in optimizer.param_groups if len(group.get("params", [])) > 0]
        if len(non_empty_groups) != len(optimizer.param_groups):
            optimizer.param_groups = non_empty_groups
        return optimizer

    def get_train_dataloader(self):
        if self.aux_train_dataset is None:
            return super().get_train_dataloader()
        # Build main dataloader normally
        dl_main = super().get_train_dataloader()
        # Build aux dataloader by temporarily swapping the train dataset
        saved = self.train_dataset
        self.train_dataset = self.aux_train_dataset
        dl_aux = super().get_train_dataloader()
        self.train_dataset = saved
        main_count = self.args.gradient_accumulation_steps - self.aux_microbatches
        return InterleavedDataLoader(dl_main, dl_aux,
                                     main_count=main_count,
                                     aux_count=self.aux_microbatches)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs.pop("num_items_in_batch", None)

        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits.detach()
        del teacher_outputs

        inputs["labels"] = labels

        # Shift for next-token prediction
        student_logits_shifted = student_logits[..., :-1, :]
        teacher_logits_shifted = teacher_logits[..., :-1, :].contiguous()
        del teacher_logits
        shift_labels = labels[..., 1:].contiguous()

        mask = (shift_labels != -100).float()

        # Compute losses
        hard_loss = compute_ce_loss(student_logits_shifted, shift_labels, mask)

        if self.alpha > 0:
            kl_loss = compute_kl_loss(
                student_logits_shifted,
                teacher_logits_shifted,
                mask,
                self.temperature
            )
        else:
            kl_loss = torch.tensor(0.0, device=mask.device)

        loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss

        # Accumulate per-component losses for correct logging
        self._loss_accum["hard_loss"] += hard_loss.detach().item()
        self._loss_accum["kl_loss"] += kl_loss.detach().item()
        self._micro_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Override to inject averaged component losses at step boundaries."""
        if self._micro_steps > 0 and "loss" in logs:
            for k, v in self._loss_accum.items():
                logs[k] = v / self._micro_steps
            self._loss_accum = {k: 0.0 for k in self._loss_accum}
            self._micro_steps = 0
        super().log(logs, start_time=start_time)


def preprocess_logits_for_metrics(logits, labels):
    """Compute predictions on GPU before moving to CPU."""
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def compute_metrics(eval_pred):
    """Compute token-level accuracy for validation."""
    predictions, labels = eval_pred
    predictions = predictions[..., :-1]
    labels = labels[..., 1:]
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / max(mask.sum(), 1)
    return {"accuracy": float(accuracy)}


def load_or_build_dataset(max_length, tokenizer, local_rank, world_size, data_cache_dir=None):
    """Load tokenized+packed dataset from cache, or build and cache it.

    data_cache_dir: directory containing (or to create) tokenized_slimpajama_packed/.
    If not given, defaults to ./data_{max_length}_cache/ relative to CWD.
    """
    if data_cache_dir is None:
        data_cache_dir = f"./data_{max_length}_cache"

    cache_path = f"{data_cache_dir}/tokenized_slimpajama_packed"
    val_cache_path = f"{data_cache_dir}/tokenized_slimpajama_val_packed"

    if os.path.exists(cache_path) and os.path.exists(val_cache_path):
        if local_rank == 0:
            print(f"  Loading cached datasets from {data_cache_dir}...")
        train_ds = load_from_disk(cache_path)
        val_ds = load_from_disk(val_cache_path)
        if local_rank == 0:
            print(f"  Loaded: {len(train_ds)} train, {len(val_ds)} val sequences")
        if world_size > 1:
            torch.distributed.barrier()
        return train_ds, val_ds

    # Build cache: only rank 0 tokenizes and packs; other ranks wait via file polling
    if local_rank == 0:
        print(f"\n  Cache not found at {data_cache_dir}, building...")
        os.makedirs(data_cache_dir, exist_ok=True)
        train_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train")
        val_dataset_raw = load_dataset("DKYoon/SlimPajama-6B", split="validation")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)

        columns_to_remove = train_dataset.column_names

        num_proc = os.cpu_count()
        print(f"  Tokenizing train (seq_len={max_length}, num_proc={num_proc})...")
        tokenized_train_raw = train_dataset.map(
            tokenize_function, batched=True, remove_columns=columns_to_remove,
            num_proc=num_proc, desc=f"Tokenizing train ({max_length})")

        print(f"  Tokenizing val (seq_len={max_length}, num_proc={num_proc})...")
        tokenized_val_raw = val_dataset_raw.map(
            tokenize_function, batched=True, remove_columns=columns_to_remove,
            num_proc=num_proc, desc=f"Tokenizing val ({max_length})")

        print("  Packing sequences...")
        train_ds = pack_sequences(tokenized_train_raw, max_length, tokenizer.eos_token_id)
        val_ds = pack_sequences(tokenized_val_raw, max_length, tokenizer.eos_token_id)
        print(f"  Packed: {len(train_ds)} train, {len(val_ds)} val sequences")

        train_ds.save_to_disk(cache_path)
        val_ds.save_to_disk(val_cache_path)
        print("  Cache saved!")
    else:
        import time
        print(f"[Rank {local_rank}] Waiting for rank 0 to prepare data (seq_len={max_length})...")
        while not (os.path.exists(cache_path) and os.path.exists(val_cache_path)):
            time.sleep(5)
        train_ds = load_from_disk(cache_path)
        val_ds = load_from_disk(val_cache_path)
        print(f"[Rank {local_rank}] Loaded from cache!")

    if world_size > 1:
        torch.distributed.barrier()

    return train_ds, val_ds


def run_training_stage(student_model, teacher_model, tokenizer, train_dataset, val_dataset,
                       output_dir, max_steps, per_device_batch_size, gradient_accumulation_steps,
                       learning_rate, warmup_steps, lr_scheduler_type, temperature, alpha,
                       local_rank, stage_name="", gradient_checkpointing=True,
                       lr_scheduler_kwargs=None, resume_from_checkpoint=False,
                       aux_train_dataset=None, aux_microbatches=2):
    """Run one stage of distillation training. Returns the trainer (model is updated in-place)."""
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"  {stage_name}")
        print(f"  Steps: {max_steps}, LR: {learning_rate}, Scheduler: {lr_scheduler_type}")
        print(f"  Batch: {per_device_batch_size} x grad_accum={gradient_accumulation_steps}")
        print(f"{'='*60}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs or {},
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        torch_compile=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        logging_first_step=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=1000,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[MemoryCleanupCallback()],
        teacher_model=teacher_model,
        temperature=temperature,
        alpha=alpha,
        max_steps=max_steps,
        aux_train_dataset=aux_train_dataset,
        aux_microbatches=aux_microbatches,
    )

    # Resume from checkpoint if requested and one exists
    ckpt = None
    if resume_from_checkpoint:
        from transformers.trainer_utils import get_last_checkpoint
        ckpt = get_last_checkpoint(output_dir)
        if ckpt and local_rank == 0:
            print(f"  Resuming from checkpoint: {ckpt}")

    trainer.train(resume_from_checkpoint=ckpt)

    # Save stage model
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    if local_rank == 0:
        print(f"  Stage model saved to: {final_model_dir}")

    return trainer


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    gradient_accumulation_steps = max(1, args.global_batch_size // (args.per_device_batch_size * world_size))

    if local_rank == 0:
        print(f"Configuration:")
        print(f"  World size: {world_size}")
        print(f"  Global batch size: {args.global_batch_size}")
        print(f"  Per-device batch size: {args.per_device_batch_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Merge KV heads: {args.merge_kv_heads}")
        if args.merge_kv_heads:
            print(f"  Num key heads: {args.num_key_heads}")
        print(f"  Train only attention: {args.train_only_attention}")
        print(f"  Knowledge Distillation:")
        print(f"    Temperature: {args.temperature}")
        print(f"    Alpha: {args.alpha}")
        if args.aux_data_cache_dir:
            print(f"  Mixed-length training:")
            print(f"    Aux data cache: {args.aux_data_cache_dir}")
            print(f"    Aux microbatches per step: {args.aux_microbatches}")
            print(f"    Main microbatches per step: {gradient_accumulation_steps - args.aux_microbatches}")

    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./experiments", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    if local_rank == 0:
        print(f"\nExperiment directory: {output_dir}")

    # Load tokenizer and models
    if local_rank == 0:
        print(f"\nLoading tokenizer and models: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if local_rank == 0:
        print("Compiling teacher model...")
    teacher_model = torch.compile(teacher_model)

    if local_rank == 0:
        print("Loading student model...")

    # Check if student is a pre-merged checkpoint (already has reduced k_proj)
    _student_already_merged = False
    if args.student_model_name != args.model_name:
        config_path = os.path.join(args.student_model_name, "config.json")
        if os.path.exists(config_path):
            import json as _json
            with open(config_path) as _f:
                _cfg = _json.load(_f)
            if _cfg.get("num_key_heads") is not None:
                _student_already_merged = True
                if local_rank == 0:
                    print(f"  Detected pre-merged student (num_key_heads={_cfg['num_key_heads']})")

    if _student_already_merged:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
        from safetensors.torch import load_file as _load_sf
        import glob as _globmod
        _ckpt_path = os.path.join(args.student_model_name, "model.safetensors")
        if os.path.exists(_ckpt_path):
            _ckpt = _load_sf(_ckpt_path, device="cpu")
        else:
            _shard_files = sorted(_globmod.glob(os.path.join(args.student_model_name, "model-*.safetensors")))
            _ckpt = {}
            for _sf in _shard_files:
                _ckpt.update(_load_sf(_sf, device="cpu"))
        for _li, _layer in enumerate(student_model.model.layers):
            _wk = f"model.layers.{_li}.self_attn.k_proj.weight"
            _bk = f"model.layers.{_li}.self_attn.k_proj.bias"
            if _wk in _ckpt:
                _w = _ckpt[_wk]
                _has_bias = _bk in _ckpt
                _new_k = torch.nn.Linear(_w.shape[1], _w.shape[0], bias=_has_bias,
                                         dtype=_layer.self_attn.k_proj.weight.dtype)
                _new_k.weight = torch.nn.Parameter(_w.to(dtype=_new_k.weight.dtype))
                if _has_bias:
                    _new_k.bias = torch.nn.Parameter(_ckpt[_bk].to(dtype=_new_k.bias.dtype))
                _layer.self_attn.k_proj = _new_k
        del _ckpt
        if local_rank == 0:
            print(f"  Loaded pre-merged k_proj weights from checkpoint")
    else:
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=torch.bfloat16)

    if args.merge_kv_heads and not _student_already_merged:
        if local_rank == 0:
            print("Merging key projection heads in student model...")
        student_model = merge_key_projection_heads(
            student_model, new_heads=args.num_key_heads, local_rank=local_rank)

    if args.train_only_attention:
        if local_rank == 0:
            print("Freezing non-attention parameters...")
        student_model = freeze_non_attention_parameters(student_model, local_rank=local_rank)

    # Pre-load all datasets upfront
    if local_rank == 0:
        print("\n--- Loading datasets ---")
    train_ds_1, val_ds_1 = load_or_build_dataset(args.max_length, tokenizer, local_rank, world_size,
                                                    data_cache_dir=args.data_cache_dir)

    # Load auxiliary dataset for mixed-length training
    aux_train_ds = None
    if args.aux_data_cache_dir:
        if local_rank == 0:
            print(f"\n--- Loading auxiliary dataset for mixed-length training ---")
        aux_train_ds, _ = load_or_build_dataset(None, tokenizer, local_rank, world_size,
                                                  data_cache_dir=args.aux_data_cache_dir)

    # --max-steps 0: only build the data cache (used by run_prepare_data.sh)
    if args.max_steps <= 0:
        if local_rank == 0:
            print(f"\nData cache ready at {args.data_cache_dir}; skipping training (--max-steps {args.max_steps}).")
        return

    # ==================== Training ====================
    os.makedirs(output_dir, exist_ok=True)

    run_training_stage(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=train_ds_1,
        val_dataset=val_ds_1,
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type else "cosine",
        temperature=args.temperature,
        alpha=args.alpha,
        local_rank=local_rank,
        stage_name="Training",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lr_scheduler_kwargs={"num_decay_steps": args.num_decay_steps} if args.num_decay_steps else None,
        resume_from_checkpoint=args.resume_from_checkpoint,
        aux_train_dataset=aux_train_ds,
        aux_microbatches=args.aux_microbatches,
    )

    if local_rank == 0:
        print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
