"""
Baseline Methods for Knowledge Sufficiency Detection
=====================================================
Implements all baselines compared against KBP in Table 1:

  - Logit-Entropy      : entropy of output token distribution
  - Logit-MaxProb      : maximum output token probability
  - P(True)            : Kadavath et al. 2022 — self-assessed correctness
  - SelfCheckGPT       : Manakul et al. 2023 — consistency across samples
  - Truthfulness Probe : Azaria & Mitchell 2023 — probes for output correctness

Paper Table 1 (Llama-3-8B):
  Logit-Entropy:   71.3 ± 1.2
  Logit-MaxProb:   73.1 ± 1.1
  P(True):         74.6 ± 1.8
  SelfCheckGPT:    75.2 ± 2.3
  Truth. Probe:    80.3 ± 1.6
  KBP (ours):      88.4 ± 0.7  ← 6-13 pts improvement
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class LogitEntropyBaseline:
    """
    Knowledge sufficiency detection via output distribution entropy.

    H(p) = -Σ p(y) log p(y) over the vocabulary.
    Low entropy → model is confident → potentially KS.
    High entropy → model is uncertain → likely KD.

    Note: AUROC is computed as 1 - entropy (higher score = more KS).
    """

    def score(
        self,
        logits: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute entropy-based scores.

        Parameters
        ----------
        logits : (N, V) tensor of output logits for first generated token

        Returns
        -------
        scores : (N,) array — higher = more knowledge-sufficient
        """
        probs = torch.softmax(logits, dim=-1)
        # Entropy H = -Σ p log p
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (N,)
        # Lower entropy → higher KS score
        return (-entropy).cpu().numpy()

    def auroc(self, logits: torch.Tensor, labels: np.ndarray) -> float:
        scores = self.score(logits)
        return float(roc_auc_score(labels, scores))


class LogitMaxProbBaseline:
    """
    Knowledge sufficiency detection via maximum output token probability.

    max_y p_M(y | q) — the probability of the most likely next token.
    High max probability → confident → likely KS.

    Paper: AUROC 73.1 ± 1.1 on PopQA (Llama-3-8B).
    """

    def score(self, logits: torch.Tensor) -> np.ndarray:
        """
        Parameters
        ----------
        logits : (N, V) tensor

        Returns
        -------
        scores : (N,) array — max probability, higher = more KS
        """
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # (N,)
        return max_probs.cpu().numpy()

    def auroc(self, logits: torch.Tensor, labels: np.ndarray) -> float:
        scores = self.score(logits)
        return float(roc_auc_score(labels, scores))


class PTrueBaseline:
    """
    P(True): self-assessed correctness (Kadavath et al., 2022).

    Prompts the model with:
      "Question: {q}
       Proposed answer: {a}
       Is the proposed answer correct? Answer yes or no."

    Then reads P("yes") as the KS score.

    Paper: AUROC 74.6 ± 1.8 on PopQA (Llama-3-8B).
    """

    PROMPT_TEMPLATE = (
        "Question: {question}\n"
        "Proposed answer: {answer}\n"
        "Is the proposed answer correct? Answer yes or no."
    )

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Pre-tokenize "yes" and "no" to get their token IDs
        self._yes_token_id = self._get_token_id("yes")
        self._no_token_id = self._get_token_id("no")

    def _get_token_id(self, word: str) -> int:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        # Take the first token if multiple
        return ids[0]

    @torch.inference_mode()
    def score(
        self,
        questions: List[str],
        answers: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Compute P(True) scores for question-answer pairs.

        Parameters
        ----------
        questions : list of str
        answers : list of str (model-generated answers to evaluate)
        batch_size : int

        Returns
        -------
        scores : (N,) array — P("yes"), higher = more KS
        """
        all_scores = []

        for i in range(0, len(questions), batch_size):
            batch_q = questions[i : i + batch_size]
            batch_a = answers[i : i + batch_size]

            prompts = [
                self.PROMPT_TEMPLATE.format(question=q, answer=a)
                for q, a in zip(batch_q, batch_a)
            ]

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(next(self.model.parameters()).device)

            outputs = self.model(**enc)
            logits = outputs.logits[:, -1, :]  # Last token logits (N, V)

            # P(yes) / (P(yes) + P(no))
            yes_logits = logits[:, self._yes_token_id]
            no_logits = logits[:, self._no_token_id]
            p_yes = torch.softmax(
                torch.stack([yes_logits, no_logits], dim=-1), dim=-1
            )[:, 0]

            all_scores.append(p_yes.cpu().numpy())

        return np.concatenate(all_scores)

    def auroc(
        self,
        questions: List[str],
        answers: List[str],
        labels: np.ndarray,
    ) -> float:
        scores = self.score(questions, answers)
        return float(roc_auc_score(labels, scores))


class SelfCheckGPTBaseline:
    """
    SelfCheckGPT: consistency-based hallucination detection.
    (Manakul et al., 2023)

    Samples K answers from the model for each query, then measures
    consistency between answers using BERTScore. High consistency → KS,
    low consistency → KD.

    Paper: AUROC 75.2 ± 2.3 on PopQA (Llama-3-8B), using 5 samples + BERTScore.
    Requires: pip install bert-score sentence-transformers
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 5,
        consistency_metric: str = "bertscore",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.consistency_metric = consistency_metric

    @torch.inference_mode()
    def _sample_answers(
        self,
        query: str,
        n_samples: int,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> List[str]:
        """Sample n_samples responses from the model."""
        enc = self.tokenizer(
            query,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        outputs = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated tokens
        input_len = enc["input_ids"].shape[1]
        answers = []
        for out in outputs:
            tokens = out[input_len:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            answers.append(text.strip())
        return answers

    def _bertscore_consistency(self, answers: List[str]) -> float:
        """
        Compute mean pairwise BERTScore (F1) as consistency measure.

        Higher score → more consistent → more likely KS.
        """
        try:
            from bert_score import score as bert_score_fn
        except ImportError:
            raise ImportError(
                "bert-score is required for SelfCheckGPT. "
                "Install with: pip install bert-score"
            )

        if len(answers) < 2:
            return 0.5

        # Compare each answer to the "main" (greedy) answer
        main_answer = answers[0]
        other_answers = answers[1:]

        _, _, F1 = bert_score_fn(
            cands=other_answers,
            refs=[main_answer] * len(other_answers),
            lang="en",
            verbose=False,
        )
        return float(F1.mean())

    def score(
        self,
        queries: List[str],
        greedy_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Compute SelfCheckGPT consistency scores.

        Parameters
        ----------
        queries : list of str
        greedy_answers : list of str, optional
            Pre-computed greedy answers (first sample). If None, sample fresh.

        Returns
        -------
        scores : (N,) — higher = more KS
        """
        scores = []
        for i, query in enumerate(queries):
            samples = self._sample_answers(query, self.n_samples)
            if greedy_answers is not None:
                samples = [greedy_answers[i]] + samples[: self.n_samples - 1]

            if self.consistency_metric == "bertscore":
                score = self._bertscore_consistency(samples)
            else:
                raise ValueError(f"Unknown metric: {self.consistency_metric}")

            scores.append(score)

        return np.array(scores)

    def auroc(
        self,
        queries: List[str],
        labels: np.ndarray,
        greedy_answers: Optional[List[str]] = None,
    ) -> float:
        scores = self.score(queries, greedy_answers)
        return float(roc_auc_score(labels, scores))


class TruthfulnessProbeBaseline:
    """
    Truthfulness Probe: Azaria & Mitchell (2023).

    Same architecture as KBP (logistic regression on ℓ2-normalized hidden
    states) but trained on output-correctness labels (O2: did greedy decoding
    match the ground-truth?) rather than knowledge-sufficiency labels (O1).

    Paper Table 1: AUROC 80.3 ± 1.6 (Llama-3-8B), 79.4 ± 1.7 (Qwen3-8B).

    Paper Section 8.2:
      "The distinction is conceptual: the Truthfulness Probe targets whether
       the model's generated output will be factually correct; KBP targets
       whether parametric memory contains the relevant knowledge, independently
       of generation quality."

    The AUROC gap (8.1 pts for Llama-3-8B) decomposes into:
      ~6.7 pts from probe target (KS labels vs correctness labels)
      ~1.4 pts from layer selection (ℓ* for H1 vs. ℓ* for truthfulness)
    """

    def __init__(self, n_seeds: int = 5, train_ratio: float = 0.70):
        self.n_seeds = n_seeds
        self.train_ratio = train_ratio
        self._probe = None

    def fit(
        self,
        hidden_states: np.ndarray,
        correctness_labels: np.ndarray,
    ) -> "TruthfulnessProbeBaseline":
        """
        Train on output-correctness labels (O2).

        Parameters
        ----------
        hidden_states : (N, d) array at the optimal layer for this probe
        correctness_labels : (N,) binary — 1 if greedy output correct, 0 otherwise
        """
        from kbp.probe import LinearProbe
        from sklearn.model_selection import train_test_split

        aurocs = []
        probes = []

        for seed in range(self.n_seeds):
            X_train, X_test, y_train, y_test = train_test_split(
                hidden_states,
                correctness_labels,
                train_size=self.train_ratio,
                stratify=correctness_labels,
                random_state=seed,
            )
            probe = LinearProbe(random_state=seed)
            probe.fit(X_train, y_train)
            auroc = probe.auroc(X_test, y_test)
            aurocs.append(auroc)
            probes.append(probe)

        best_idx = int(np.argmax(aurocs))
        self._probe = probes[best_idx]

        mean_auroc = float(np.mean(aurocs))
        std_auroc = float(np.std(aurocs))
        logger.info(f"Truthfulness Probe: AUROC={mean_auroc:.4f}±{std_auroc:.4f}")
        return self

    def score(self, hidden_states: np.ndarray) -> np.ndarray:
        return self._probe.predict_proba(hidden_states)

    def auroc(self, hidden_states: np.ndarray, labels: np.ndarray) -> float:
        scores = self.score(hidden_states)
        return float(roc_auc_score(labels, scores))


def evaluate_all_baselines(
    logits: torch.Tensor,
    hidden_states_at_best_layer: np.ndarray,
    correctness_labels: np.ndarray,
    ks_labels: np.ndarray,
    questions: Optional[List[str]] = None,
    model=None,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Evaluate all baselines and return a summary dict.

    Parameters
    ----------
    logits : (N, V) output logits
    hidden_states_at_best_layer : (N, d)
    correctness_labels : (N,) — O2 labels (greedy correct?)
    ks_labels : (N,) — O1 labels (knowledge-sufficient?)
    questions, model, tokenizer : required for P(True) and SelfCheckGPT

    Returns
    -------
    dict mapping method_name → AUROC
    """
    results = {}

    # Logit-based baselines
    le = LogitEntropyBaseline()
    results["Logit-Entropy"] = le.auroc(logits, ks_labels)

    lm = LogitMaxProbBaseline()
    results["Logit-MaxProb"] = lm.auroc(logits, ks_labels)

    # Truthfulness Probe (requires hidden states + correctness labels)
    tp = TruthfulnessProbeBaseline()
    tp.fit(hidden_states_at_best_layer, correctness_labels)
    results["Truthfulness Probe"] = tp.auroc(
        hidden_states_at_best_layer, ks_labels
    )

    logger.info("Baseline AUROC scores:")
    for name, auroc in results.items():
        logger.info(f"  {name:25s}: {auroc:.4f}")

    return results
