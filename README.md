# Knowledge Boundary Probe (KBP)

> **"Does the Model Know What It Knows?"**  
> Detecting Knowledge Boundaries via Internal Representation Geometry

A PyTorch implementation of the Knowledge Boundary Probe (KBP), a lightweight detection head that identifies whether an LLM query falls within or outside the model's parametric knowledge — **at inference time, with <1% latency overhead**.

---

## Overview

LLMs confabulate with the same fluency as they recall. KBP addresses this by probing mid-layer hidden states to detect the *geometric signature* of knowledge deficiency before generation occurs.

### Two core hypotheses (both confirmed):

| Hypothesis | Claim | Result |
|---|---|---|
| **H1** | Knowledge sufficiency is **linearly separable** in mid-layer hidden states | AUROC 0.86–0.88, +6–8 pts over baselines |
| **H2** | Knowledge-deficient queries show higher **activation effective rank** (label-free) | ρs ≈ −0.82 with gradient SNR |

### Applications

- **Dynamic RAG Routing**: Retrieve only when the model lacks knowledge (51% retrieval rate vs 100%, matching full-retrieval accuracy)
- **Fine-tuning Investment Decision**: Predict SFT viability *before* data collection via label-free effective rank

---

## Installation

```bash
git clone https://github.com/your-org/kbp.git
cd kbp
pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0, transformers ≥ 4.40

---

## Quick Start

### 1. Supervised KBP (with labeled pilot data)

```python
from kbp import KBP

# Initialize with a model
kbp = KBP.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Train the probe on PopQA (or any labeled set)
kbp.fit(train_queries, train_labels)  # labels: 1=sufficient, 0=deficient

# Inference
result = kbp.predict("Who wrote the novel 1984?")
print(result.label)   # "KNOWLEDGE_SUFFICIENT"
print(result.score)   # 0.93
print(result.margin)  # 1.87  ← prompt sensitivity indicator
```

### 2. Unsupervised KBP (label-free, effective rank)

```python
from kbp import KBP

kbp = KBP.from_pretrained("meta-llama/Meta-Llama-3-8B", mode="unsupervised")

# Calibrate on 256 unlabeled queries from target domain
kbp.calibrate(pilot_queries)

result = kbp.predict("What is the GDP of Laos in 2019?")
print(result.label)   # "KNOWLEDGE_DEFICIENT"
print(result.erank)   # 41.3
```

### 3. Dynamic RAG Routing

```python
from kbp.routing import KBPRouter

router = KBPRouter(kbp, retriever=your_retriever)

answer = router.answer("Who is the current president of France?")
# Automatically retrieves only when KBP signals knowledge deficiency
```

---

## Reproducing Paper Results

### Step 1: Extract hidden states

```bash
python scripts/extract_hidden_states.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset popqa \
    --output data/hidden_states/llama3_popqa.pt \
    --layers 16 32
```

### Step 2: Train probe and evaluate (H1)

```bash
python experiments/run_h1.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --model llama3 \
    --seeds 5
```

Expected output (Table 1 in paper):
```
KBP Linear Probe (best layer, O1)  AUROC: 88.4 ± 0.7  [Layer 23]
Truthfulness Probe (baseline)       AUROC: 80.3 ± 1.6
SelfCheckGPT (baseline)            AUROC: 75.2 ± 2.3
```

### Step 3: Effective rank analysis (H2)

```bash
python experiments/run_h2.py \
    --model meta-llama/Meta-Llama-3-8B \
    --tasks popqa_high popqa_mid popqa_low medbench laobench mkqa_arabic \
    --n-queries 256
```

### Step 4: Cross-domain transfer

```bash
python experiments/run_transfer.py \
    --probe-checkpoint checkpoints/kbp_llama3_popqa.pkl \
    --eval-datasets medbench laobench mmlu mkqa
```

---

## Architecture

```
Query q
   │
   ▼
LLM Forward Pass (layers 1→L)
   │
   ├─── Layer hook at ℓ* (60–75% depth)
   │         │
   │         ▼
   │    h(ℓ*)(q) ∈ ℝᵈ
   │         │
   │    ℓ2-Normalize → h̃
   │         │
   │    ┌────┴────┐
   │    │         │
   │  Supervised  Unsupervised
   │  s=σ(wᵀh̃+b)  r=erank(C(ℓ*))
   │    │         │
   │  s≥0.5: KS  r<τ_lo: KS
   │  s<0.5: KD  r>τ_hi: KD
   │             else: Uncertain
   │
   ▼
Dynamic RAG Routing  /  SFT Investment Decision
```

---

## Project Structure

```
kbp/
├── kbp/
│   ├── extractor.py        # Hidden state extraction with layer hooks
│   ├── probe.py            # Linear probe (logistic regression)
│   ├── effective_rank.py   # Effective rank computation + H2 analysis
│   ├── kbp.py             # Main KBP pipeline (supervised + unsupervised)
│   ├── routing.py         # Dynamic RAG router
│   ├── baselines.py       # Logit-Entropy, MaxProb, P(True), SelfCheckGPT
│   └── metrics.py         # AUROC, gradient SNR, Spearman correlation
├── scripts/
│   ├── extract_hidden_states.py
│   ├── train_probe.py
│   ├── evaluate_baselines.py
│   └── compute_effective_rank.py
├── experiments/
│   ├── run_h1.py          # H1: linear separability experiment
│   ├── run_h2.py          # H2: activation dispersion experiment
│   ├── run_transfer.py    # Cross-domain transfer
│   └── run_routing.py     # RAG routing evaluation (Table 4)
├── configs/
│   └── default.yaml
├── tests/
│   ├── test_extractor.py
│   ├── test_probe.py
│   └── test_effective_rank.py
└── docs/
    └── theory.md
```

---

## Key Results

### Table 1: Knowledge Sufficiency Detection (PopQA)

| Method | Llama-3-8B | Qwen3-8B |
|--------|-----------|---------|
| Random baseline | 50.0 | 50.0 |
| Logit-Entropy | 71.3±1.2 | 70.6±1.3 |
| Logit-MaxProb | 73.1±1.1 | 72.4±1.2 |
| P(True) | 74.6±1.8 | 73.9±1.9 |
| SelfCheckGPT | 75.2±2.3 | 74.5±2.4 |
| Truthfulness Probe | 80.3±1.6 | 79.4±1.7 |
| **KBP Linear Probe** | **88.4±0.7** | **86.3±0.8** |

### Table 4: RAG Routing (PopQA low+mid frequency)

| Strategy | Acc. (%) | % Retrieved | Latency |
|----------|----------|------------|---------|
| Never retrieve | 41.3 | 0% | 1.00× |
| Always retrieve | 67.4 | 100% | 2.73× |
| KBP (unsupervised) | 64.2 | 57% | **1.94×** |
| KBP (supervised) | **66.9** | **51%** | **1.83×** |

---

## Citation

```bibtex
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
