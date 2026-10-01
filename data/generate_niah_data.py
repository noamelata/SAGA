#!/usr/bin/env python3
import argparse
import os
import string

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Word pools for generating diverse content
# ---------------------------------------------------------------------------

# Common English words for distractor text
DISTRACTOR_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "great", "between", "need", "large", "often",
    "around", "each", "long", "while", "before", "right", "still", "small",
    "number", "place", "found", "study", "much", "through", "line", "system",
    "set", "under", "group", "however", "general", "part", "during", "process",
    "important", "without", "form", "state", "point", "home", "same", "another",
    "world", "house", "different", "school", "might", "development", "government",
    "research", "information", "national", "report", "according", "analysis",
    "evidence", "following", "provide", "several", "program", "result",
    "question", "available", "service", "although", "community", "public",
    "market", "company", "technology", "performance", "experience", "management",
]

# Key categories — different domains to add variety
KEY_DOMAINS = {
    "city": ["Tokyo", "London", "Paris", "Berlin", "Sydney", "Cairo", "Mumbai",
             "Toronto", "Seoul", "Bangkok", "Lima", "Oslo", "Dublin", "Prague",
             "Vienna", "Lisbon", "Warsaw", "Athens", "Helsinki", "Budapest",
             "Bucharest", "Zagreb", "Tallinn", "Riga", "Vilnius", "Bratislava",
             "Ljubljana", "Reykjavik", "Bern", "Brussels", "Amsterdam", "Copenhagen",
             "Stockholm", "Ankara", "Nairobi", "Bogota", "Santiago", "Havana",
             "Manila", "Jakarta", "Hanoi", "Dhaka", "Islamabad", "Riyadh"],
    "color": ["crimson", "azure", "emerald", "amber", "violet", "coral",
              "indigo", "scarlet", "turquoise", "magenta", "ivory", "obsidian",
              "burgundy", "sapphire", "olive", "maroon", "teal", "lavender",
              "charcoal", "bronze", "copper", "platinum", "ruby", "pearl"],
    "animal": ["falcon", "dolphin", "panther", "cobra", "eagle", "wolf",
               "tiger", "whale", "raven", "fox", "hawk", "lynx", "otter",
               "bison", "crane", "viper", "jaguar", "puma", "heron", "ibis",
               "condor", "osprey", "marten", "badger", "ferret", "stoat"],
    "element": ["hydrogen", "helium", "lithium", "carbon", "nitrogen", "oxygen",
                "neon", "sodium", "silicon", "iron", "copper", "silver",
                "gold", "mercury", "uranium", "titanium", "platinum", "cobalt",
                "nickel", "zinc", "argon", "krypton", "xenon", "radium"],
    "fruit": ["pomegranate", "dragonfruit", "starfruit", "passionfruit", "lychee",
              "guava", "persimmon", "kumquat", "papaya", "mango", "durian",
              "jackfruit", "rambutan", "tamarind", "plantain", "quince",
              "mulberry", "gooseberry", "boysenberry", "elderberry"],
    "code": None,  # generated randomly
}

# Templates for embedding key-value pairs
KV_TEMPLATES = [
    # Prose style
    "The {category} assigned to {key} is {value}.",
    "For the record, {key} corresponds to {value}.",
    "According to the database, the entry for {key} is {value}.",
    "It has been determined that {key} maps to {value}.",
    "The official designation for {key} is {value}.",
    "Records indicate that {key} is associated with {value}.",
    "In this context, {key} refers to {value}.",
    "The identifier linked to {key} is {value}.",
    # Report style
    "Entry {key}: the registered value is {value}.",
    "Item {key} — classification: {value}.",
    "Reference {key} has been catalogued as {value}.",
    "Lookup result for {key}: {value}.",
    # Conversational
    "Note that {key} has the value {value}.",
    "Remember, the answer for {key} is {value}.",
    "As documented, {key} yields {value}.",
]

# Templates for queries
QUERY_TEMPLATES_SINGLE = [
    "Question: What is the value assigned to {key}?\nAnswer: {value}",
    "Query: What does {key} map to?\nAnswer: {value}",
    "What is the entry for {key}? The answer is {value}.",
    "Look up {key}. Result: {value}.",
    "Retrieve the value for {key}. Answer: {value}.",
]

QUERY_TEMPLATES_MULTI = [
    "Question: What are the values for {keys}?\nAnswer: {values}",
    "Query: Look up the entries for {keys}.\nAnswer: {values}",
    "Retrieve the values associated with {keys}. Results: {values}.",
    "What do {keys} map to? The answers are {values}.",
    "Find the designations for {keys}. Answer: {values}",
]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_code_value(rng):
    """Generate a random alphanumeric code."""
    length = rng.integers(6, 12)
    chars = string.ascii_uppercase + string.digits
    return "".join(rng.choice(list(chars)) for _ in range(length))


def generate_key_value_pairs(rng, num_keys):
    """Generate N unique key-value pairs from random domains."""
    domains = list(KEY_DOMAINS.keys())
    pairs = []
    used_keys = set()
    used_values = set()

    for _ in range(num_keys):
        domain = rng.choice(domains)
        pool = KEY_DOMAINS[domain]

        if pool is None:  # code domain
            key = f"ID-{generate_code_value(rng)}"
            value = generate_code_value(rng)
        else:
            # Pick unused key and value
            available_keys = [k for k in pool if k not in used_keys]
            if len(available_keys) < 2:
                # Fall back to code
                key = f"ID-{generate_code_value(rng)}"
                value = generate_code_value(rng)
            else:
                key = rng.choice(available_keys)
                # Value from a different domain
                other_domain = rng.choice([d for d in domains if d != domain and KEY_DOMAINS[d] is not None])
                other_pool = KEY_DOMAINS[other_domain]
                available_values = [v for v in other_pool if v not in used_values]
                if available_values:
                    value = rng.choice(available_values)
                else:
                    value = generate_code_value(rng)

        used_keys.add(key)
        used_values.add(value)
        pairs.append((key, value))

    return pairs


def generate_distractor_sentence(rng, min_words=8, max_words=25):
    """Generate a random distractor sentence."""
    length = rng.integers(min_words, max_words + 1)
    words = [rng.choice(DISTRACTOR_WORDS) for _ in range(length)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def generate_distractor_paragraph(rng, min_sentences=3, max_sentences=8):
    """Generate a paragraph of distractor text."""
    n = rng.integers(min_sentences, max_sentences + 1)
    sentences = [generate_distractor_sentence(rng) for _ in range(n)]
    return " ".join(sentences)


def generate_example_text(rng, num_keys, max_length, tokenizer):
    """Generate one complete retrieval example as text, tokenize-checked to fill max_length.

    Structure:
    - Distractor paragraphs with key-value pairs scattered throughout
    - Query at the end asking for the values
    """
    pairs = generate_key_value_pairs(rng, num_keys)

    # Build the KV insertion sentences
    kv_sentences = []
    for key, value in pairs:
        template = rng.choice(KV_TEMPLATES)
        kv_sentences.append(template.format(key=key, value=value, category="entry"))

    # Build query (at the end)
    if num_keys == 1:
        template = rng.choice(QUERY_TEMPLATES_SINGLE)
        query = template.format(key=pairs[0][0], value=pairs[0][1])
    else:
        keys_str = ", ".join(k for k, v in pairs)
        values_str = ", ".join(f"{k}: {v}" for k, v in pairs)
        template = rng.choice(QUERY_TEMPLATES_MULTI)
        query = template.format(keys=keys_str, values=values_str)

    # Tokenize the query to know how much space we need to reserve
    query_tokens = tokenizer.encode("\n\n" + query, add_special_tokens=False)
    body_budget = max_length - len(query_tokens) - 10  # small margin

    # Generate body: interleave distractor paragraphs with KV sentences
    # First, decide KV positions (which paragraph index to insert after)
    n_target_paragraphs = rng.integers(15, 30)  # enough to fill 4K tokens
    kv_positions = sorted(rng.choice(range(1, n_target_paragraphs), size=num_keys, replace=False))

    body_parts = []
    kv_idx = 0
    para_idx = 0

    while True:
        # Generate a distractor paragraph
        para = generate_distractor_paragraph(rng, min_sentences=4, max_sentences=10)
        body_parts.append(para)

        # Insert KV pair if scheduled at this position
        if kv_idx < num_keys and para_idx in kv_positions:
            body_parts.append(kv_sentences[kv_idx])
            kv_idx += 1

        para_idx += 1

        # Check if we have enough tokens
        body_text = "\n\n".join(body_parts)
        body_tokens = tokenizer.encode(body_text, add_special_tokens=False)
        if len(body_tokens) >= body_budget:
            break

    # Insert any remaining KV pairs
    while kv_idx < num_keys:
        body_parts.insert(-1, kv_sentences[kv_idx])
        kv_idx += 1

    # Combine body and query, ensuring query is preserved
    body_text = "\n\n".join(body_parts)
    body_tokens = tokenizer.encode(body_text, add_special_tokens=False)
    query_text = "\n\n" + query
    query_tokens_final = tokenizer.encode(query_text, add_special_tokens=False)

    # Truncate body to make room for query
    max_body = max_length - len(query_tokens_final)
    if len(body_tokens) > max_body:
        body_tokens = body_tokens[:max_body]

    tokens = body_tokens + query_tokens_final
    tokens = tokens[:max_length]
    return tokens


def _worker_init(tokenizer_path, max_length):
    """Initialize per-worker tokenizer (avoid pickling)."""
    global _worker_tokenizer, _worker_max_length
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _worker_max_length = max_length


def _worker_generate(task):
    """Generate one example in a worker process."""
    seed, num_keys = task
    rng = np.random.default_rng(seed)
    return generate_example_text(rng, num_keys, _worker_max_length, _worker_tokenizer)


def generate_dataset(args):
    """Generate the full synthetic dataset."""
    import multiprocessing as mp

    print(f"Generating {args.num_examples} synthetic NIAH examples")
    print(f"  Max length: {args.max_length} tokens")
    print(f"  Output: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  Workers: {args.num_workers}")

    rng = np.random.default_rng(args.seed)

    # Key distribution: weighted toward multikey (2-4 keys)
    # This targets the multikey_2 and multikey_3 ruler subtasks
    key_weights = {1: 0.10, 2: 0.25, 3: 0.30, 4: 0.25, 5: 0.10}
    key_choices = list(key_weights.keys())
    key_probs = np.array([key_weights[k] for k in key_choices])
    key_probs = key_probs / key_probs.sum()

    # Pre-generate all task parameters (seed per example + num_keys)
    per_example_seeds = rng.integers(0, 2**62, size=args.num_examples)
    num_keys_all = rng.choice(key_choices, size=args.num_examples, p=key_probs)
    tasks = list(zip(per_example_seeds.tolist(), num_keys_all.tolist()))

    key_dist = {k: 0 for k in key_choices}
    for nk in num_keys_all:
        key_dist[nk] += 1

    print(f"\n  Key distribution:")
    for k in sorted(key_dist.keys()):
        print(f"    {k} keys: {key_dist[k]} ({100*key_dist[k]/args.num_examples:.1f}%)")

    # Parallel generation
    print(f"\n  Generating with {args.num_workers} workers...")
    with mp.Pool(args.num_workers, initializer=_worker_init,
                 initargs=(args.tokenizer, args.max_length)) as pool:
        all_input_ids = list(tqdm(
            pool.imap(_worker_generate, tasks, chunksize=100),
            total=args.num_examples, desc="  Generating", unit="ex"))

    # Build dataset
    print(f"\n  Building Arrow dataset...")
    ones_row = [1] * args.max_length
    ds = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": [ones_row] * len(all_input_ids),
        "labels": all_input_ids,
    })

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tokenized_niah_synthetic")
    print(f"  Saving to {out_path}...")
    ds.save_to_disk(out_path)
    print(f"  Saved: {len(ds)} examples of {args.max_length} tokens")

    # Spot check: decode a few examples
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"\n  === Spot check (last 400 chars of 3 random examples) ===")
    spot_idxs = rng.choice(len(all_input_ids), size=3, replace=False)
    for idx in spot_idxs:
        text = tokenizer.decode(all_input_ids[idx][-150:])
        print(f"\n  Example {idx}:")
        print(f"  ...{text[-400:]}")

    print(f"\nDone! Synthetic NIAH data at {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NIAH retrieval data")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--num-examples", type=int, default=50000,
                        help="Number of examples to generate (default: 50000)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()
    if args.num_workers is None:
        import multiprocessing as mp
        args.num_workers = mp.cpu_count()
    generate_dataset(args)


if __name__ == "__main__":
    main()
