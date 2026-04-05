# Theoretical Background

This document provides the mathematical foundations for the Knowledge Boundary Probe (KBP).

---

## 1. Knowledge Sufficiency — Formal Definition

**Definition 1 (Knowledge Sufficiency, O1 — Frequency Proxy)**

Let $\mathcal{M}$ be a language model and $q$ a factual query. Let $f(q)$ denote the Wikipedia monthly page-view count of the entity $q$ is about. We define:

$$\text{KS}(q) = \begin{cases} 1 & f(q) \ge f^* \\ 0 & f(q) < f^* \end{cases}$$

where $f^* = 10^4$ (monthly views). This is the **frequency proxy** (O1) used for probe training.

**Definition 2 (Knowledge Sufficiency, O2 — Accuracy Oracle)**

$$\text{KS}(q) = \mathbf{1}[\text{greedy}(\mathcal{M}, q) = y_q]$$

where $y_q$ is the ground-truth answer. O2 is used for the Truthfulness Probe baseline and for cross-domain evaluation.

**Paper Section 3.2** discusses why O1 is preferred for probe training:
> "O1 is strictly upstream of generation: it reflects whether the model *has* the knowledge, whereas O2 measures whether the knowledge is *accessible under greedy decoding*."

---

## 2. H1: Linear Separability of Knowledge Sufficiency

**Hypothesis H1**: For a sufficiently deep layer $\ell^*$, knowledge-sufficient (KS) and knowledge-deficient (KD) queries are *linearly separable* in the hidden-state space $\mathbb{R}^d$.

### 2.1 Probe Formulation

Let $h^{(\ell)}(q) \in \mathbb{R}^d$ be the last-token hidden state at layer $\ell$. Define the $\ell_2$-normalized representation:

$$\tilde{h}^{(\ell)}(q) = \frac{h^{(\ell)}(q)}{\|h^{(\ell)}(q)\|_2}$$

The KBP linear probe predicts:

$$s(q) = \sigma(\mathbf{w}^\top \tilde{h}^{(\ell^*)}(q) + b) \in [0, 1]$$

where $(\mathbf{w}, b)$ are learned via logistic regression and $s(q) \ge 0.5$ predicts KS.

### 2.2 Margin and Prompt Sensitivity

The geometric distance of query $q$ to the decision hyperplane $\{\mathbf{w}^\top \tilde{h} + b = 0\}$ is:

$$\delta(q) = \frac{|\mathbf{w}^\top \tilde{h}^{(\ell^*)}(q) + b|}{\|\mathbf{w}\|}$$

**Proposition 1** (Boundary Sensitivity, Section 6.3): The detection variance $\text{Var}_{q' \sim \Pi(q)}[s(q')]$ across paraphrases $q' \sim \Pi(q)$ is monotonically decreasing in $\delta(q)$.

*Sketch*: Paraphrases trace a compact neighborhood $\mathcal{N}(q)$ in $\mathbb{R}^d$. When $\delta(q)$ is small, $\mathcal{N}(q)$ straddles the hyperplane, causing the probe to flip. When $\delta(q)$ is large, $\mathcal{N}(q)$ lies entirely on one side.

### 2.3 Optimal Layer Selection

**Proposition 2** (Layer Optimality, Section 6.1): $\ell^*$ falls in the range $[0.60L, 0.75L]$, where $L$ is the total number of layers.

*Intuition*: Early layers encode syntactic and positional features; late layers encode task-completion distributions. The middle-to-late transition region (≈65% depth) is where the model's *belief about what it knows* is most salient before generation.

---

## 3. H2: Effective Rank as a Label-Free Knowledge Proxy

**Hypothesis H2**: Knowledge-deficient tasks induce higher-dimensional activation patterns than knowledge-sufficient tasks, measurable via the **effective rank** of the hidden-state covariance matrix.

### 3.1 Effective Rank

Given a matrix $H \in \mathbb{R}^{N \times d}$ of centered hidden states, let $C = \frac{1}{N} H^\top H$ be the covariance matrix with eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0$.

The **effective rank** (Roy & Vetterli, 2007) is:

$$\text{erank}(C) = \frac{\text{tr}(C)}{\|C\|_2} = \frac{\sum_i \lambda_i}{\lambda_1}$$

This equals 1 when all variance is concentrated on a single axis, and approaches $d$ when variance is uniform across all axes.

### 3.2 Geometric Interpretation

- **KS tasks** (e.g., PopQA High-frequency): The model retrieves knowledge via a small set of coherent weight-column directions. Hidden states cluster in a **low-rank submanifold**. erank ≈ 18–22.

- **KD tasks** (e.g., MedBench, LaoBench): The model falls back to statistical co-occurrence patterns without coherent retrieval. Hidden states diffuse across the full weight space. erank ≈ 38–50.

### 3.3 Connection to Gradient SNR

**Theorem 1** (Informal, Section 5.3): In the early training regime ($t \le 5$ steps), the gradient signal-to-noise ratio satisfies:

$$\text{SNR}_\mathcal{T} \propto \frac{1}{\text{erank}(C^{(\ell^*)})}$$

where the proportionality holds up to task-dependent constants. This is confirmed empirically with Spearman $\rho_s \approx -0.82$ across 12 tasks × 2 models.

*Sketch*: High erank → the gradient $\nabla \mathcal{L}$ aggregates contributions from many incoherent weight directions → the sample-to-sample gradient variance $\text{Var}[\nabla \mathcal{L}_t]$ is large → low SNR.

---

## 4. Unsupervised Threshold Calibration

Given a reference pair $(H_\text{KS}^\text{ref}, H_\text{KD}^\text{ref})$ from, e.g., PopQA high/low frequency:

$$r_{\text{ref},\text{hi}} = \text{erank}(C(H_\text{KS}^\text{ref})), \quad r_{\text{ref},\text{lo}} = \text{erank}(C(H_\text{KD}^\text{ref}))$$

The calibration thresholds are:

$$\tau_\text{lo} = r_{\text{ref},\text{hi}} + 0.4 \cdot (r_{\text{ref},\text{lo}} - r_{\text{ref},\text{hi}})$$
$$\tau_\text{hi} = r_{\text{ref},\text{hi}} + 0.7 \cdot (r_{\text{ref},\text{lo}} - r_{\text{ref},\text{hi}})$$

A new task is classified as:

$$\text{label} = \begin{cases} \text{KS} & r < \tau_\text{lo} \\ \text{KD} & r > \tau_\text{hi} \\ \text{UNCERTAIN} & \tau_\text{lo} \le r \le \tau_\text{hi} \end{cases}$$

---

## 5. Cross-Model Transfer via Procrustes Alignment

Given a probe $(\mathbf{w}, b)$ trained on model $\mathcal{M}_\text{src}$ and a set of $N_\text{align}$ unlabeled alignment queries $\{q_i\}$, the optimal rotation $R^*$ is:

$$R^* = \arg\min_{R^\top R = I} \sum_{i=1}^{N_\text{align}} \left\| \tilde{h}_\text{tgt}^{(\ell^*)}(q_i) - R\, \tilde{h}_\text{src}^{(\ell^*)}(q_i) \right\|_2^2$$

This has a closed-form solution via SVD of $H_\text{tgt}^\top H_\text{src} = U \Sigma V^\top$:

$$R^* = UV^\top$$

The aligned probe applies weight vector $\mathbf{w}_\text{aligned} = (R^*)^\top \mathbf{w}$ to target-model hidden states.

**Convergence**: AUROC stabilizes at $N_\text{align} \approx 150$ queries (Figure 8).

---

## 6. RAG Routing Expected Value

**Proposition 3** (Retrieval Decision, Appendix K): Under cost ratio $c_r / \Delta a$, retrieval has non-negative expected value when:

$$\text{AUROC}_\text{KBP} \ge \frac{1}{2} + \frac{c_r}{2\,\Delta a}$$

With $c_r / \Delta a = 0.5$, the threshold is AUROC ≥ 0.75. KBP exceeds this at **all** tested layers in the range $[0.50L, 0.90L]$.

---

## References

- Roy, O. & Vetterli, M. (2007). *The effective rank: A measure of effective dimensionality*. EUSIPCO.
- Kadavath, S. et al. (2022). *Language models (mostly) know what they know*. arXiv:2207.05221.
- Manakul, P. et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*. EMNLP.
- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. EMNLP Findings.
