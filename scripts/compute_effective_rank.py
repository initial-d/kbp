"""
Compute effective rank for one or more tasks (H2 analysis).

Outputs effective rank estimates, regime classifications, and SFT viability
decisions — without requiring any labeled data.

Usage
-----
# Single task from pre-extracted hidden states:
python scripts/compute_effective_rank.py \
    --hidden-states data/hidden_states/llama3_medbench.pt \
    --best-layer 23 \
    --task-name MedBench

# Batch: all tasks in a directory:
python scripts/compute_effective_rank.py \
    --hidden-states-dir data/hidden_states/ \
    --best-layer 23 \
    --output results/erank_table.json

# Live extraction from a model (no pre-extracted states):
python scripts/compute_effective_rank.py \
    --model meta-llama/Meta-Llama-3-8B \
    --queries data/medbench_queries.txt \
    --best-layer 23 \
    --task-name MedBench \
    --n-queries 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Compute activation effective rank")

    # Input: pre-extracted hidden states
    p.add_argument("--hidden-states", default=None,
                   help="Path to a single .pt hidden states file")
    p.add_argument("--hidden-states-dir", default=None,
                   help="Directory of .pt files (one per task)")
    p.add_argument("--best-layer", type=int, default=23,
                   help="ℓ* — layer to extract erank from")

    # Input: live extraction
    p.add_argument("--model", default=None,
                   help="HuggingFace model for live extraction")
    p.add_argument("--queries", default=None,
                   help="Path to .txt file with one query per line")
    p.add_argument("--n-queries", type=int, default=256,
                   help="Number of queries for erank estimation")

    # Task metadata
    p.add_argument("--task-name", default="Unknown Task")
    p.add_argument("--reference-ks-hs", default=None,
                   help="Reference KS hidden states for threshold calibration")
    p.add_argument("--reference-kd-hs", default=None,
                   help="Reference KD hidden states for threshold calibration")

    # Output
    p.add_argument("--output", default=None, help="Path to save JSON results")
    p.add_argument("--n-bootstrap", type=int, default=20)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def load_hidden_states(path: str, layer: int) -> np.ndarray:
    """Load and return hidden states for a single layer as numpy array."""
    from kbp.extractor import HiddenStateOutput
    hs = HiddenStateOutput.load(path)
    if layer not in hs.layer_indices:
        raise ValueError(
            f"Layer {layer} not in file {path}. Available: {hs.layer_indices}"
        )
    return hs.get_layer(layer).numpy()


def extract_live(model_name: str, queries: List[str], layer: int) -> np.ndarray:
    """Extract hidden states from a live model."""
    from kbp.extractor import HiddenStateExtractor, ExtractionConfig
    config = ExtractionConfig(layer_start=layer, layer_end=layer)
    extractor = HiddenStateExtractor(model_name, config=config)
    output = extractor.extract(queries, layers=[layer])
    return output.get_layer(layer).numpy()


def analyze_single_task(
    H: np.ndarray,
    task_name: str,
    n_bootstrap: int = 20,
    tau_lo: Optional[float] = None,
    tau_hi: Optional[float] = None,
    verbose: bool = False,
) -> Dict:
    """Run full effective rank analysis for a single task."""
    from kbp.effective_rank import (
        EffectiveRankEstimator,
        ERANK_FT_THRESHOLD,
        classify_sft_viability,
    )

    estimator = EffectiveRankEstimator(n_queries=min(256, len(H)), n_bootstrap=n_bootstrap)
    result = estimator.estimate(H)

    # Unsupervised regime classification
    if tau_lo is not None and tau_hi is not None:
        label, erank = estimator.predict_unsupervised(H, tau_lo, tau_hi)
    else:
        label = result.regime
        erank = result.erank

    # SFT viability
    sft_info = classify_sft_viability(result.erank, tau_ft=ERANK_FT_THRESHOLD)

    # Convergence check: compare N=128 vs N=256
    convergence = estimator.convergence_analysis(
        H, n_values=[64, 128, 256], n_bootstrap=5
    )

    output = {
        "task": task_name,
        "n_queries": len(H),
        "hidden_dim": H.shape[1],
        "erank": float(result.erank),
        "erank_std": float(result.erank_std),
        "regime": result.regime,
        "unsupervised_label": label,
        "sft_recommendation": sft_info["recommendation"],
        "sft_rationale": sft_info["rationale"],
        "convergence": {str(n): {"mean": v[0], "std": v[1]} for n, v in convergence.items()},
    }

    if verbose:
        # Top eigenvalues
        output["top10_eigenvalues"] = result.eigenvalues[:10].tolist()

    return output


def print_result_table(results: List[Dict]) -> None:
    """Pretty-print effective rank results."""
    print(f"\n{'═' * 75}")
    print(f"  Activation Effective Rank Analysis")
    print(f"{'═' * 75}")
    header = (
        f"{'Task':<28} {'erank':>8} {'±std':>6} "
        f"{'Regime':>8} {'SFT':>22}"
    )
    print(header)
    print("─" * 75)

    regime_order = {"KS": 0, "Mixed": 1, "KD": 2}
    for res in sorted(results, key=lambda r: (regime_order.get(r["regime"], 3), r["erank"])):
        sft = "✓ SFT viable" if res["sft_recommendation"] == "SFT_VIABLE" else "✗ Need more data"
        print(
            f"{res['task']:<28} {res['erank']:>8.2f} {res['erank_std']:>5.2f} "
            f"  {res['regime']:>8}  {sft:>22}"
        )
    print("─" * 75)
    print(f"  τ_lo={22.0} (KS ceiling), τ_hi={34.0} (KD floor), τ_ft={38.0} (SFT viability)")
    print()


def main():
    args = parse_args()

    results = []

    # ── Calibrate thresholds if reference states provided ─────────────────────
    tau_lo, tau_hi = None, None
    if args.reference_ks_hs and args.reference_kd_hs:
        from kbp.effective_rank import EffectiveRankEstimator
        logger.info("Calibrating thresholds from reference states...")
        H_ks = load_hidden_states(args.reference_ks_hs, args.best_layer)
        H_kd = load_hidden_states(args.reference_kd_hs, args.best_layer)
        est = EffectiveRankEstimator(n_bootstrap=args.n_bootstrap)
        tau_lo, tau_hi = est.calibrate_thresholds(H_ks, H_kd)
        logger.info(f"Calibrated: τ_lo={tau_lo:.2f}, τ_hi={tau_hi:.2f}")

    # ── Single task ───────────────────────────────────────────────────────────
    if args.hidden_states:
        H = load_hidden_states(args.hidden_states, args.best_layer)
        result = analyze_single_task(
            H[: args.n_queries],
            task_name=args.task_name,
            n_bootstrap=args.n_bootstrap,
            tau_lo=tau_lo,
            tau_hi=tau_hi,
            verbose=args.verbose,
        )
        results.append(result)

    # ── Batch from directory ──────────────────────────────────────────────────
    elif args.hidden_states_dir:
        hs_dir = Path(args.hidden_states_dir)
        hs_files = list(hs_dir.glob(f"*_layer{args.best_layer}.npy"))
        if not hs_files:
            hs_files = list(hs_dir.glob("*.pt"))

        logger.info(f"Found {len(hs_files)} hidden state files in {hs_dir}")
        for hs_file in sorted(hs_files):
            task_name = hs_file.stem.replace(f"_layer{args.best_layer}", "").replace("_", " ")
            try:
                if hs_file.suffix == ".npy":
                    H = np.load(hs_file)[: args.n_queries]
                else:
                    H = load_hidden_states(str(hs_file), args.best_layer)[: args.n_queries]

                result = analyze_single_task(
                    H, task_name=task_name,
                    n_bootstrap=args.n_bootstrap,
                    tau_lo=tau_lo, tau_hi=tau_hi,
                    verbose=args.verbose,
                )
                results.append(result)
                logger.info(
                    f"  {task_name}: erank={result['erank']:.2f} "
                    f"({result['regime']})"
                )
            except Exception as e:
                logger.error(f"  Error processing {hs_file}: {e}")

    # ── Live extraction ───────────────────────────────────────────────────────
    elif args.model and args.queries:
        with open(args.queries) as f:
            queries = [ln.strip() for ln in f if ln.strip()][: args.n_queries]
        logger.info(f"Extracting from model {args.model} ({len(queries)} queries)...")
        H = extract_live(args.model, queries, args.best_layer)
        result = analyze_single_task(
            H, task_name=args.task_name,
            n_bootstrap=args.n_bootstrap,
            tau_lo=tau_lo, tau_hi=tau_hi,
            verbose=args.verbose,
        )
        results.append(result)

    else:
        print("Provide one of: --hidden-states, --hidden-states-dir, or --model + --queries")
        sys.exit(1)

    # ── Print and save ────────────────────────────────────────────────────────
    print_result_table(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
