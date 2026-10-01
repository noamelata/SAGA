# More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding<br><sub>Official PyTorch implementation of SAGA (Sparse Asymmetric Group-Query Attention)</sub>

[Noam Elata](https://noamelata.github.io/)\*, [Itay Lamprecht](https://github.com/itlamp)\*, [Mikey Shechter](https://github.com/MikeyShechter)\*, [Daniel Ohayon](https://github.com/Danielohayon), [Itay Hubara](https://itayhubara.github.io/), [Daniel Soudry](https://soudry.github.io/)<br>
NeurIPS 2026

\* Equal contribution

### Abstract
*Autoregressive generation in Large Language Models (LLMs) is constrained by the memory and computational demands of attention mechanisms. Sparse attention methods mitigate this cost by selecting only high-probability entries of the attention matrix. We observe that in many such methods, this renders the probability-value multiplication negligible, shifting the bottleneck to the query-key step. Key heads can therefore be reduced to accelerate inference, while retaining more value heads preserves capacity with limited additional decoding cost. We introduce Sparse Asymmetric Group-Query Attention (SAGA), which decouples key and value head counts to exploit this principle, and pair it with approximate top-N (Atop-N) attention, a simple sparse attention method designed to study the interaction between sparsity and head-count asymmetry. We formalize the benefits of this asymmetry theoretically and validate them empirically through latency measurements and quality evaluations on models up to 1.5B parameters. Together, SAGA and Atop-N achieve end-to-end decoding speedups exceeding 2× over our full-attention GQA baseline at long contexts. Models trained from scratch with SAGA nearly match the quality of comparable GQA variants on the evaluated benchmarks. To facilitate adoption, we introduce an efficient fine-tuning method that converts pretrained models to the SAGA architecture, enabling practitioners to benefit from our approach without costly retraining.*

## Code Overview

| File | Paper Section |
|---|---|
| `gqa_distill.py` | Sec 4 (GQA→SAGA conversion), Sec 5.3 / App C.3 (distillation) |
| `merge_heads.py` | Sec 4.1 (matching-based student init) |
| `speedup.py` | Sec 5.1 (latency benchmarks) |
| `lm_eval_saga.py` | Sec 5.3, App C.4 (accuracy evaluation) |
| `attention/attention.py` | Sec 3.1 / App C.1 (all attention implementations) |
| `attention/triton_atopn.py` | App C.1 (threshold selection kernel) |
| `data/prepare_mixed_data.py` | App C.3 (training data mix) |
| `data/generate_niah_data.py` | App C.3 (synthetic NIAH data) |
| `train_scratch.py` | Sec 5.2 / App C.2 (SmolLM2-360M from scratch) |

Throughout the code and CLI flags, Atop-N is written `atopn`.

## Getting Started

Install the dependencies:

```bash
pip install -r requirements.txt
```

The repository supports three workflows. Each has a one-line entry point below and a detailed guide.

### 1. Training SAGA from scratch (Sec 5.2) — [guide](train_scratch.md)

Trains SmolLM2-360M with fewer key heads than value heads:

```bash
bash train_scratch.sh
```

### 2. Converting a pretrained model to SAGA (Secs 4, 5.3) — [guide](finetune.md)

Builds the data caches, initializes the student by merging key heads, then distills it from the original model in three stages of increasing context length:

```bash
bash run_prepare_data.sh   # build data caches (once per model)
bash finetune.sh           # student init + stages 1-3
```

### 3. Evaluation (Secs 5.1, 5.3) — [guide](eval.md)

Runs accuracy (HellaSwag, WikiText, RULER) and decoding-latency benchmarks:

```bash
EVAL_MODEL=./experiments/qwen_merged/final_model bash run_eval.sh
```

## Citation

```bibtex
@inproceedings{elata2026saga,
  title     = {More Value per Key: Asymmetric Sparse Attention for Faster {LLM} Decoding},
  author    = {Elata, Noam and Lamprecht, Itay and Shechter, Mikey and Ohayon, Daniel and Hubara, Itay and Soudry, Daniel},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
