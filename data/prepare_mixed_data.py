#!/usr/bin/env python3
import argparse
import os
import shutil

import numpy as np
from datasets import Dataset, load_from_disk, load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Dataset format handlers
# ---------------------------------------------------------------------------

def load_and_format_texts(path):
    """Load a dataset from disk or HF hub and convert to plain text strings.

    Auto-detects format from column names:
      - text:                   use directly
      - messages:               join role/content pairs
      - instruction/input/output: concatenate
      - context/question/answer: QA template
      - article/question/options/answer: QuALITY template
    """
    # Try loading from local directory first, then as HF dataset name
    if os.path.isdir(path):
        ds = load_dataset("json", data_dir=path, split="train")
    else:
        try:
            ds = load_dataset(path, split="train")
        except Exception:
            for subset in ["fulltext", "distractor", "squad_v2", "all", None]:
                try:
                    ds = load_dataset(path, subset, split="train")
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Could not load dataset '{path}'. "
                                 "Pass a local path or ensure the dataset is in the HF cache.")

    cols = set(ds.column_names)
    texts = []

    if "messages" in cols:
        # LongAlign: list of {role, content} dicts
        for row in ds:
            parts = []
            for msg in row["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            texts.append("\n".join(parts))

    elif "instruction" in cols and "output" in cols:
        # LongAlpaca: instruction / input / output
        for row in ds:
            parts = [row.get("instruction", "")]
            inp = row.get("input", "")
            if inp:
                parts.append(inp)
            parts.append(row.get("output", ""))
            texts.append("\n".join(parts))

    elif "article" in cols and "options" in cols:
        # QuALITY: article / question / options / answer
        for row in ds:
            article = row.get("article", "")
            question = row.get("question", "")
            options = row.get("options", [])
            answer_idx = row.get("answer", 0)
            if isinstance(options, list) and len(options) > 0:
                try:
                    answer_text = options[int(answer_idx)]
                except (IndexError, ValueError, TypeError):
                    answer_text = str(answer_idx)
            else:
                answer_text = str(answer_idx)
            texts.append(f"{article}\nQuestion: {question}\nAnswer: {answer_text}")

    elif "context" in cols and "question" in cols:
        # SQuAD / HotpotQA style
        for row in ds:
            context = row.get("context", "")
            question = row.get("question", "")
            # Handle both 'answer' (str) and 'answers' (dict with 'text' list)
            answer = row.get("answer", "")
            if not answer:
                answers = row.get("answers", {})
                if isinstance(answers, dict):
                    answer_list = answers.get("text", [])
                    answer = answer_list[0] if answer_list else ""
                elif isinstance(answers, list) and len(answers) > 0:
                    answer = answers[0]
            texts.append(f"{context}\nQuestion: {question}\nAnswer: {answer}")

    elif "text" in cols:
        # FineWeb-edu or generic text
        texts = ds["text"]

    else:
        raise ValueError(f"Unrecognized columns in {path}: {cols}")

    print(f"    {path}: {len(texts)} examples loaded")
    return texts


# ---------------------------------------------------------------------------
# Token-level mixing
# ---------------------------------------------------------------------------

def texts_to_flat_tokens(texts, tokenizer, eos_token_id, batch_size=1000):
    """Tokenize a list of texts and concatenate with EOS separators into a flat numpy array.

    Uses batch tokenization for efficiency.
    """
    from tqdm import tqdm

    # Filter empty texts upfront
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return np.array([], dtype=np.int32)

    # Batch tokenize
    all_ids = []
    for start in tqdm(range(0, len(texts), batch_size), desc="    Tokenizing", unit="batch"):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        all_ids.extend(encoded)

    # Concatenate with EOS separators
    total_tokens = sum(len(ids) for ids in all_ids) + max(0, len(all_ids) - 1)
    pieces = np.empty(total_tokens, dtype=np.int32)
    pos = 0
    for i, ids in enumerate(all_ids):
        if i > 0:
            pieces[pos] = eos_token_id
            pos += 1
        pieces[pos:pos + len(ids)] = ids
        pos += len(ids)

    return pieces[:pos]


def texts_to_padded_chunks(texts, tokenizer, eos_token_id, max_length, batch_size=1000):
    """Tokenize texts individually, pad to max_length, skip examples that exceed max_length.

    Returns a Dataset with input_ids, attention_mask, labels. Padding positions have
    attention_mask=0 and labels=-100 so they are ignored in loss computation.
    """
    from tqdm import tqdm

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return None, 0, 0

    # Batch tokenize
    all_ids = []
    for start in tqdm(range(0, len(texts), batch_size), desc="    Tokenizing", unit="batch"):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        all_ids.extend(encoded)

    # Filter: skip examples > max_length, pad shorter ones
    kept_input_ids = []
    kept_attention_mask = []
    kept_labels = []
    n_skipped = 0
    for ids in all_ids:
        if len(ids) > max_length:
            n_skipped += 1
            continue
        pad_len = max_length - len(ids)
        kept_input_ids.append(ids + [eos_token_id] * pad_len)
        kept_attention_mask.append([1] * len(ids) + [0] * pad_len)
        kept_labels.append(ids + [-100] * pad_len)

    if not kept_input_ids:
        return None, 0, n_skipped

    ds = Dataset.from_dict({
        "input_ids": kept_input_ids,
        "attention_mask": kept_attention_mask,
        "labels": kept_labels,
    })
    return ds, n_skipped, len(kept_input_ids)


def build_mixed_cache(args):
    unpacked_ratio = getattr(args, 'unpacked_ratio', 0.0)
    unpacked_datasets = getattr(args, 'unpacked_datasets', None) or []

    sp_fraction = 1.0 - args.extra_ratio - args.niah_ratio - unpacked_ratio
    print(f"Building mixed data cache")
    print(f"  Output: {args.output_dir}")
    print(f"  SP ratio: {sp_fraction:.2%}")
    print(f"  Extra ratio (packed): {args.extra_ratio:.2%}")
    print(f"  Unpacked ratio: {unpacked_ratio:.2%}")
    print(f"  NIAH ratio: {args.niah_ratio}")
    print(f"  Max length: {args.max_length}")
    print(f"  Seed: {args.seed}")

    if sp_fraction <= 0:
        raise ValueError(f"SP fraction is {sp_fraction:.2%} — ratios sum to >= 1.0")

    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # --- Load existing SlimPajama packed cache ---
    print(f"\nLoading SlimPajama cache from {args.slimpajama_cache}...")
    sp_ds = load_from_disk(args.slimpajama_cache)
    n_sp_chunks = len(sp_ds)
    n_sp_tokens = n_sp_chunks * args.max_length
    print(f"  SlimPajama: {n_sp_tokens:,} tokens ({n_sp_chunks} chunks of {args.max_length})")

    datasets_to_combine = [sp_ds]

    # --- Load and tokenize packed extra datasets ---
    n_extra_chunks = 0
    if args.extra_datasets and args.extra_ratio > 0:
        print(f"\nLoading and tokenizing {len(args.extra_datasets)} packed extra datasets...")
        extra_token_arrays = []
        for i, ds_path in enumerate(args.extra_datasets, 1):
            print(f"\n  [{i}/{len(args.extra_datasets)}] {ds_path}")
            texts = load_and_format_texts(ds_path)
            tokens = texts_to_flat_tokens(texts, tokenizer, eos_token_id)
            print(f"      -> {len(tokens):,} tokens")
            if len(tokens) > 0:
                extra_token_arrays.append(tokens)

        if extra_token_arrays:
            all_extra = np.concatenate(extra_token_arrays)
            del extra_token_arrays
            print(f"\n  Total packed extra tokens: {all_extra.shape[0]:,}")

            target_extra = int(n_sp_tokens * args.extra_ratio / sp_fraction)
            print(f"  Target packed extra tokens (for {args.extra_ratio:.0%} mix): {target_extra:,}")

            if len(all_extra) >= target_extra:
                all_extra = all_extra[:target_extra]
                print(f"  Truncated packed extra tokens to {target_extra:,}")
            else:
                n_repeats = (target_extra // len(all_extra)) + 1
                all_extra = np.tile(all_extra, n_repeats)[:target_extra]
                print(f"  Repeated packed extra tokens to reach {target_extra:,}")

            n_extra_chunks = len(all_extra) // args.max_length
            all_extra = all_extra[:n_extra_chunks * args.max_length]
            extra_chunked = all_extra.reshape(n_extra_chunks, args.max_length)
            del all_extra
            print(f"  Packed extra: {n_extra_chunks} chunks")

            ones_row = [1] * args.max_length
            extra_ds = Dataset.from_dict({
                "input_ids": extra_chunked.tolist(),
                "attention_mask": [ones_row] * n_extra_chunks,
                "labels": extra_chunked.tolist(),
            })
            del extra_chunked
            datasets_to_combine.append(extra_ds)

    # --- Load and tokenize unpacked datasets (padded, skip > max_length) ---
    n_unpacked_chunks = 0
    if unpacked_datasets and unpacked_ratio > 0:
        print(f"\nLoading {len(unpacked_datasets)} unpacked datasets (pad to {args.max_length}, skip longer)...")
        unpacked_ds_list = []
        total_kept = 0
        total_skipped = 0
        for i, ds_path in enumerate(unpacked_datasets, 1):
            print(f"\n  [{i}/{len(unpacked_datasets)}] {ds_path}")
            texts = load_and_format_texts(ds_path)
            ds_result, n_skipped, n_kept = texts_to_padded_chunks(
                texts, tokenizer, eos_token_id, args.max_length)
            if ds_result is not None:
                unpacked_ds_list.append(ds_result)
                total_kept += n_kept
                total_skipped += n_skipped
                print(f"      -> {n_kept} kept, {n_skipped} skipped (>{args.max_length} tokens)")
            else:
                total_skipped += n_skipped
                print(f"      -> 0 kept, {n_skipped} skipped")

        if unpacked_ds_list:
            from datasets import concatenate_datasets as concat_ds
            unpacked_combined = concat_ds(unpacked_ds_list)
            del unpacked_ds_list

            target_unpacked = int(n_sp_chunks * unpacked_ratio / sp_fraction)
            print(f"\n  Total unpacked: {len(unpacked_combined)} chunks ({total_skipped} skipped)")
            print(f"  Target unpacked chunks (for {unpacked_ratio:.0%} mix): {target_unpacked}")

            if len(unpacked_combined) >= target_unpacked:
                # Shuffle then select
                unpacked_combined = unpacked_combined.shuffle(seed=args.seed)
                unpacked_combined = unpacked_combined.select(range(target_unpacked))
                print(f"  Sampled down to {target_unpacked} chunks")
            else:
                # Repeat to reach target
                n_repeats = (target_unpacked // len(unpacked_combined)) + 1
                indices = list(range(len(unpacked_combined))) * n_repeats
                unpacked_combined = unpacked_combined.select(indices[:target_unpacked])
                print(f"  Repeated unpacked to reach {target_unpacked} chunks")

            n_unpacked_chunks = len(unpacked_combined)
            datasets_to_combine.append(unpacked_combined)

    # --- Load NIAH dataset if specified ---
    n_niah_chunks = 0

    if args.niah_dataset and args.niah_ratio > 0:
        print(f"\nLoading NIAH dataset from {args.niah_dataset}...")
        niah_ds = load_from_disk(args.niah_dataset)
        target_niah_chunks = int(n_sp_chunks * args.niah_ratio / sp_fraction)
        print(f"  NIAH available: {len(niah_ds)} chunks")
        print(f"  NIAH target: {target_niah_chunks} chunks")

        if len(niah_ds) >= target_niah_chunks:
            niah_ds = niah_ds.select(range(target_niah_chunks))
        else:
            # Repeat to reach target
            n_repeats = (target_niah_chunks // len(niah_ds)) + 1
            indices = list(range(len(niah_ds))) * n_repeats
            niah_ds = niah_ds.select(indices[:target_niah_chunks])

        # Pad short rows to max_length (NIAH generator may produce slightly short examples)
        def _pad_row(row):
            pad_id = eos_token_id
            for key in ["input_ids", "labels"]:
                if len(row[key]) < args.max_length:
                    row[key] = row[key] + [pad_id] * (args.max_length - len(row[key]))
            if len(row["attention_mask"]) < args.max_length:
                row["attention_mask"] = row["attention_mask"] + [0] * (args.max_length - len(row["attention_mask"]))
            return row

        niah_ds = niah_ds.map(_pad_row)
        print(f"  NIAH using: {len(niah_ds)} chunks (padded to {args.max_length})")
        n_niah_chunks = len(niah_ds)
        datasets_to_combine.append(niah_ds)

    total_chunks = n_sp_chunks + n_extra_chunks + n_unpacked_chunks + n_niah_chunks
    parts_str = f"{n_sp_chunks} SP"
    if n_extra_chunks:
        parts_str += f" + {n_extra_chunks} packed_extra"
    if n_unpacked_chunks:
        parts_str += f" + {n_unpacked_chunks} unpacked"
    if n_niah_chunks:
        parts_str += f" + {n_niah_chunks} NIAH"
    print(f"\nCombining: {parts_str} = {total_chunks} total")

    # --- Concatenate and shuffle at the Arrow level (no row iteration) ---
    from datasets import concatenate_datasets
    print(f"  Concatenating datasets...")
    combined_ds = concatenate_datasets(datasets_to_combine)
    del datasets_to_combine

    print(f"  Shuffling {total_chunks} chunks...")
    combined_ds = combined_ds.shuffle(seed=args.seed)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tokenized_slimpajama_packed")
    print(f"  Saving train cache to {out_path}...")
    combined_ds.save_to_disk(out_path)
    print(f"  Saved: {len(combined_ds)} train chunks of {args.max_length} tokens")

    # Copy val set from original cache
    _copy_val_set(args)

    print(f"\nDone! Mixed cache at {args.output_dir}")


def _copy_val_set(args):
    """Copy the validation set from the original SlimPajama cache directory."""
    sp_cache_dir = os.path.dirname(args.slimpajama_cache)
    val_src = os.path.join(sp_cache_dir, "tokenized_slimpajama_val_packed")
    val_dst = os.path.join(args.output_dir, "tokenized_slimpajama_val_packed")

    if os.path.exists(val_src):
        if os.path.exists(val_dst):
            shutil.rmtree(val_dst)
        shutil.copytree(val_src, val_dst)
        print(f"  Copied val set from {val_src}")
    else:
        print(f"  WARNING: val set not found at {val_src} — you may need to copy it manually")


def main():
    parser = argparse.ArgumentParser(description="Build mixed training data cache")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer (e.g. ./models/Qwen2.5-1.5B)")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Sequence length for packed chunks (default: 4096)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for mixed cache")
    parser.add_argument("--slimpajama-cache", type=str, required=True,
                        help="Path to existing SlimPajama packed cache "
                             "(e.g. ./data_4096_cache/tokenized_slimpajama_packed)")
    parser.add_argument("--extra-datasets", type=str, nargs="*", default=[],
                        help="Paths to extra datasets (local dirs or HF hub names)")
    parser.add_argument("--extra-ratio", type=float, default=0.10,
                        help="Fraction of final data from packed extra sources (default: 0.10)")
    parser.add_argument("--unpacked-datasets", type=str, nargs="*", default=[],
                        help="Paths to long-context datasets to include unpacked "
                             "(each example padded to max-length, longer examples skipped)")
    parser.add_argument("--unpacked-ratio", type=float, default=0.0,
                        help="Fraction of final data from unpacked sources (default: 0.0)")
    parser.add_argument("--niah-dataset", type=str, default=None,
                        help="Path to pre-tokenized NIAH Arrow dataset "
                             "(e.g. ./data_niah_synthetic/tokenized_niah_synthetic)")
    parser.add_argument("--niah-ratio", type=float, default=0.0,
                        help="Fraction of final data from NIAH dataset (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    build_mixed_cache(args)


if __name__ == "__main__":
    main()
