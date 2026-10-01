import os
import argparse
import json
import torch
import networkx as nx

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data-driven KV head merging for student initialization. "
                    "Halves num_key_value_heads by finding optimal head pairs via "
                    "minimum-weight matching on key activation statistics."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-cache-dir", type=str, required=True,
                        help="Path to tokenized_slimpajama_packed/ directory.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-batches", type=int, default=1,
                        help="Number of data batches for activation collection.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--edge-weight", type=str, default="score_l2",
                        choices=["average", "score_l2"],
                        help="'average': Frobenius distance between K activations. "
                             "'score_l2': closed-form score-matching objective (default).")
    return parser.parse_args()


def _collate_fn(batch):
    return {k: torch.stack([torch.tensor(x[k]) for x in batch]) for k in batch[0]}


@torch.no_grad()
def _collect_attention_stats(model, dataset, n_batches, batch_size, max_length,
                              collect_query_grams=False):
    config = model.config
    num_kv_heads = config.num_key_value_heads
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    q_group = num_heads // num_kv_heads

    input_device = next(model.parameters()).device
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    n_layers = len(model.model.layers)
    per_layer_chunks = [[] for _ in range(n_layers)]
    per_layer_query_grams = [None] * n_layers

    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        input_ids = batch["input_ids"].to(input_device)[:, :max_length]
        attention_mask = batch["attention_mask"].to(input_device)[:, :max_length]

        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states

        for layer_idx, layer in enumerate(model.model.layers):
            X = hidden_states[layer_idx]
            Wk = layer.self_attn.k_proj.weight.view(num_kv_heads, head_dim, hidden_size)
            K = torch.einsum("bsh,kdh->bskd", X, Wk)
            B, S = K.shape[:2]
            K = K.reshape(B * S, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
            per_layer_chunks[layer_idx].append(K.cpu())

            if collect_query_grams:
                Wq = layer.self_attn.q_proj.weight.view(num_heads, head_dim, hidden_size)
                Q = torch.einsum("bsh,qdh->bsqd", X, Wq)
                Q = Q.reshape(B * S, num_heads, head_dim).permute(1, 0, 2)
                Q = Q.view(num_kv_heads, q_group, B * S, head_dim).float()
                grams = torch.einsum("gqtd,gqte->gde", Q, Q).cpu()
                per_layer_query_grams[layer_idx] = (
                    grams if per_layer_query_grams[layer_idx] is None
                    else per_layer_query_grams[layer_idx] + grams
                )

    per_layer_A = [torch.cat(chunks, dim=1) for chunks in per_layer_chunks]
    return per_layer_A, per_layer_query_grams


def _score_l2_pair_merge(key_acts_a, key_acts_b, query_gram_a, query_gram_b, wk_a, wk_b):
    """Closed-form optimal merge weight minimizing the score-matching objective."""
    total_gram = query_gram_a + query_gram_b
    pinv = torch.linalg.pinv(total_gram)
    merged_wk = pinv @ (query_gram_a @ wk_a + query_gram_b @ wk_b)
    merged_acts = (key_acts_a @ query_gram_a + key_acts_b @ query_gram_b) @ pinv
    delta_a = merged_acts - key_acts_a
    delta_b = merged_acts - key_acts_b
    cost = (((delta_a @ query_gram_a) * delta_a).sum()
            + ((delta_b @ query_gram_b) * delta_b).sum()) / key_acts_a.shape[-1]
    return merged_wk, cost


@torch.no_grad()
def merge_heads(model, dataset, n_batches, batch_size, max_length, edge_weight):
    """Merge KV heads pairwise using data-driven minimum-weight matching.

    Halves num_key_value_heads. Reorders Q, V, O projections to match new grouping.
    Returns (model, all_pairs).
    """
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    q_group = num_heads // num_kv_heads
    new_kv_heads = num_kv_heads // 2

    assert num_kv_heads % 2 == 0, "num_key_value_heads must be even (pairwise merge)"

    per_layer_A, per_layer_query_grams = _collect_attention_stats(
        model, dataset, n_batches, batch_size, max_length,
        collect_query_grams=(edge_weight == "score_l2"),
    )

    all_pairs = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        k_proj, q_proj, v_proj, o_proj = attn.k_proj, attn.q_proj, attn.v_proj, attn.o_proj
        dev = k_proj.weight.device

        A = per_layer_A[layer_idx].to(device=dev, dtype=torch.float32)
        Wk = k_proj.weight.view(num_kv_heads, head_dim, hidden_size).float()

        graph = nx.Graph()
        graph.add_nodes_from(range(num_kv_heads))

        if edge_weight == "average":
            A_flat = A.reshape(num_kv_heads, -1)
            G = A_flat @ A_flat.t()
            n2 = (A_flat * A_flat).sum(dim=1, keepdim=True)
            dist = torch.sqrt(torch.clamp(n2 + n2.t() - 2.0 * G, min=0.0))
            dist.fill_diagonal_(float("inf"))
            for i in range(num_kv_heads):
                for j in range(i + 1, num_kv_heads):
                    graph.add_edge(i, j, weight=float(dist[i, j].cpu()))
            matching = nx.min_weight_matching(graph)
            pairs = sorted(sorted(e) for e in matching)
            pair_idx = torch.tensor(pairs, dtype=torch.long, device=dev)
            new_Wk = Wk[pair_idx].mean(dim=1)
        else:
            query_grams = per_layer_query_grams[layer_idx].to(device=dev, dtype=torch.float32)
            merged_by_pair = {}
            for i in range(num_kv_heads):
                for j in range(i + 1, num_kv_heads):
                    mwk, cost = _score_l2_pair_merge(
                        A[i], A[j], query_grams[i], query_grams[j], Wk[i], Wk[j])
                    merged_by_pair[(i, j)] = mwk
                    graph.add_edge(i, j, weight=cost.item())
            matching = nx.min_weight_matching(graph)
            pairs = sorted(sorted(e) for e in matching)
            new_Wk = torch.stack([merged_by_pair[(a, b)] for a, b in pairs])

        all_pairs.append(pairs)
        new_Wk = new_Wk.reshape(new_kv_heads * head_dim, hidden_size)
        k_proj.weight = torch.nn.Parameter(
            new_Wk.to(dtype=k_proj.weight.dtype), requires_grad=True)
        if k_proj.bias is not None:
            # Merge biases with the same pairing (average, as in gqa_distill.merge_key_projection_heads)
            bk = k_proj.bias.view(num_kv_heads, head_dim).float()
            pair_idx = torch.tensor(pairs, dtype=torch.long, device=dev)
            k_proj.bias = torch.nn.Parameter(
                bk[pair_idx].mean(dim=1).reshape(new_kv_heads * head_dim).to(k_proj.bias.dtype))

        # Permute Q, V, O to stay consistent with new KV grouping
        q_perm = torch.tensor(
            [idx for a, b in pairs
             for idx in list(range(a * q_group, (a + 1) * q_group))
                       + list(range(b * q_group, (b + 1) * q_group))],
            dtype=torch.long, device=dev)
        kv_perm = torch.tensor(
            [idx for a, b in pairs for idx in [a, b]],
            dtype=torch.long, device=dev)

        q_w = q_proj.weight.view(num_heads, head_dim, hidden_size)
        q_proj.weight = torch.nn.Parameter(
            q_w[q_perm].reshape(num_heads * head_dim, hidden_size).to(q_proj.weight.dtype))
        if q_proj.bias is not None:
            q_proj.bias = torch.nn.Parameter(
                q_proj.bias.view(num_heads, head_dim)[q_perm]
                .reshape(num_heads * head_dim).to(q_proj.bias.dtype))

        o_w = o_proj.weight.view(hidden_size, num_heads, head_dim)
        o_proj.weight = torch.nn.Parameter(
            o_w[:, q_perm, :].reshape(hidden_size, num_heads * head_dim).to(o_proj.weight.dtype))

        v_w = v_proj.weight.view(num_kv_heads, head_dim, hidden_size)
        v_proj.weight = torch.nn.Parameter(
            v_w[kv_perm].reshape(num_kv_heads * head_dim, hidden_size).to(v_proj.weight.dtype))
        if v_proj.bias is not None:
            v_proj.bias = torch.nn.Parameter(
                v_proj.bias.view(num_kv_heads, head_dim)[kv_perm]
                .reshape(num_kv_heads * head_dim).to(v_proj.bias.dtype))

    return model, all_pairs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    dataset = load_from_disk(args.data_cache_dir)

    model, all_pairs = merge_heads(
        model=model,
        dataset=dataset,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        max_length=args.max_length,
        edge_weight=args.edge_weight,
    )

    model.config.num_key_heads = model.config.num_key_value_heads // 2
    model.config.kv_merge_pairs = all_pairs
    with open(os.path.join(args.output_dir, "kv_merge_pairs.json"), "w") as f:
        json.dump(all_pairs, f)

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Student saved to {args.output_dir}  (num_key_heads={model.config.num_key_heads})")
    print(f"Layer 0 pairs: {all_pairs[0]}")


if __name__ == "__main__":
    main()
