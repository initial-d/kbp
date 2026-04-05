"""
Evaluate all baseline methods against KBP on a held-out dataset.

Reproduces Table 1 from the paper (AUROC comparison).

Usage
-----
python scripts/evaluate_baselines.py \
    --hidden-states data/hidden_states/llama3_popqa.pt \
    --logits data/logits/llama3_popqa.pt \
    --probe-checkpoint checkpoints/kbp_llama3_layer23.pkl \
    --best-layer 23 \
    --output results/table1.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all baselines (Table 1)")
    p.add_argument("--hidden-states", required=True,
                   help="Path to .pt hidden states file")
    p.add_argument("--logits", default=None,
                   help="Path to .pt logits file (N, V) for logit-based baselines")
    p.add_argument("--probe-checkpoint", required=True,
                   help="Path to trained KBP probe .pkl")
    p.add_argument("--best-layer", type=int, default=23)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--output", default="results/table1.json")
    p.add_argument("--model-name", default="Llama-3-8B")
    return p.parse_args()


def main():
    args = parse_args()

    from kbp.extractor import HiddenStateOutput
    from kbp.probe import LinearProbe
    from sklearn.model_selection import train_test_split

    # ── Load hidden states and labels ─────────────────────────────────────────
    logger.info(f"Loading hidden states from {args.hidden_states}")
    hs_output = HiddenStateOutput.load(args.hidden_states)
    H = hs_output.get_layer(args.best_layer).numpy()

    label_path = Path(args.hidden_states).with_suffix(".labels.npy")
    if not label_path.exists():
        logger.error(f"Labels not found: {label_path}")
        sys.exit(1)
    ks_labels = np.load(label_path)  # O1: knowledge-sufficiency labels

    # Correctness labels (O2) — may or may not be available
    correctness_path = Path(args.hidden_states).with_suffix(".correctness.npy")
    if correctness_path.exists():
        correctness_labels = np.load(correctness_path)
        logger.info("O2 (correctness) labels loaded")
    else:
        logger.warning(
            "Correctness labels (O2) not found. "
            "Truthfulness Probe and P(True) will be skipped."
        )
        correctness_labels = None

    # Logits (for Logit-Entropy / Logit-MaxProb)
    logits = None
    if args.logits and Path(args.logits).exists():
        import torch
        logits = torch.load(args.logits, map_location="cpu")
        logger.info(f"Logits loaded: shape {logits.shape}")

    # ── Fixed train/test split (seed 0, to match H1 evaluation) ─────────────
    X_train, X_test, y_train, y_test = train_test_split(
        H, ks_labels,
        train_size=args.train_ratio,
        stratify=ks_labels,
        random_state=0,
    )
    n_train, n_test = len(X_train), len(X_test)
    logger.info(f"Train: {n_train}, Test: {n_test}")

    results = {}

    # ── 1. KBP Linear Probe ───────────────────────────────────────────────────
    logger.info("Evaluating KBP Linear Probe...")
    probe = LinearProbe.load(args.probe_checkpoint)
    kbp_auroc = probe.auroc(X_test, y_test)
    results["KBP Linear Probe"] = {"auroc": kbp_auroc, "std": None}
    logger.info(f"  KBP: AUROC={kbp_auroc:.4f}")

    # Multi-seed for std
    aurocs_kbp = []
    for seed in range(args.n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            H, ks_labels, train_size=args.train_ratio,
            stratify=ks_labels, random_state=seed,
        )
        p = LinearProbe(random_state=seed)
        p.fit(X_tr, y_tr)
        aurocs_kbp.append(p.auroc(X_te, y_te))
    results["KBP Linear Probe"]["auroc"] = float(np.mean(aurocs_kbp))
    results["KBP Linear Probe"]["std"] = float(np.std(aurocs_kbp))
    logger.info(
        f"  KBP (multi-seed): AUROC={np.mean(aurocs_kbp):.4f}±{np.std(aurocs_kbp):.4f}"
    )

    # ── 2. Truthfulness Probe (O2 labels) ─────────────────────────────────────
    if correctness_labels is not None:
        logger.info("Evaluating Truthfulness Probe (O2 labels)...")
        from kbp.baselines import TruthfulnessProbeBaseline
        tp = TruthfulnessProbeBaseline(n_seeds=args.n_seeds, train_ratio=args.train_ratio)
        tp.fit(H, correctness_labels)
        tp_auroc = tp.auroc(X_test, y_test)
        results["Truthfulness Probe"] = {"auroc": tp_auroc}
        logger.info(f"  Truth. Probe: AUROC={tp_auroc:.4f}")

    # ── 3. Logit-based baselines ──────────────────────────────────────────────
    if logits is not None:
        from kbp.baselines import LogitEntropyBaseline, LogitMaxProbBaseline

        # Use same test split by index
        test_idx = np.where(np.isin(np.arange(len(H)), np.where(
            np.isin(H, X_test).all(axis=1)
        )))[0][:n_test]
        # simpler: split logits the same way by using the same random seed
        logits_train, logits_test = train_test_split(
            logits, train_size=args.train_ratio, random_state=0
        )

        logger.info("Evaluating Logit-Entropy baseline...")
        le = LogitEntropyBaseline()
        le_auroc = le.auroc(logits_test, y_test)
        results["Logit-Entropy"] = {"auroc": float(le_auroc)}
        logger.info(f"  Logit-Entropy: AUROC={le_auroc:.4f}")

        logger.info("Evaluating Logit-MaxProb baseline...")
        lm = LogitMaxProbBaseline()
        lm_auroc = lm.auroc(logits_test, y_test)
        results["Logit-MaxProb"] = {"auroc": float(lm_auroc)}
        logger.info(f"  Logit-MaxProb: AUROC={lm_auroc:.4f}")

    # ── 4. Random baseline ────────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(42)
    random_auroc = float(roc_auc_score(y_test, rng.rand(len(y_test))))
    results["Random"] = {"auroc": random_auroc}

    # ── Print Table 1 ─────────────────────────────────────────────────────────
    paper_aurocs = {
        "Random":             (50.0, 0.0),
        "Logit-Entropy":      (71.3, 1.2),
        "Logit-MaxProb":      (73.1, 1.1),
        "P(True)":            (74.6, 1.8),
        "SelfCheckGPT":       (75.2, 2.3),
        "Truthfulness Probe": (80.3, 1.6),
        "KBP Linear Probe":   (88.4, 0.7),
    }

    print(f"\n{'═' * 70}")
    print(f"  Table 1: Knowledge Sufficiency Detection AUROC — {args.model_name}")
    print(f"{'═' * 70}")
    print(f"{'Method':<26} {'Ours':>10} {'Paper':>10}")
    print("─" * 70)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["auroc"],
    )
    for name, res in sorted_results:
        auroc = res["auroc"] * 100
        std = res.get("std")
        std_str = f"±{std*100:.1f}" if std is not None else ""
        paper = paper_aurocs.get(name)
        paper_str = f"{paper[0]:.1f}±{paper[1]:.1f}" if paper else "—"
        print(f"{name:<26} {auroc:>7.1f}{std_str:>5}  {paper_str:>10}")

    print(f"\n  KBP improvement over best baseline:")
    kbp_auroc_pct = results["KBP Linear Probe"]["auroc"] * 100
    best_baseline_auroc = max(
        v["auroc"] for k, v in results.items() if k != "KBP Linear Probe"
    ) * 100
    print(f"  +{kbp_auroc_pct - best_baseline_auroc:.1f} AUROC points "
          f"({kbp_auroc_pct:.1f} vs {best_baseline_auroc:.1f})")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "best_layer": args.best_layer,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
