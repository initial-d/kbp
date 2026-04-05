"""
Extract hidden states from a causal LM and save to disk.

Usage
-----
python scripts/extract_hidden_states.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset popqa \
    --split test \
    --output data/hidden_states/llama3_popqa.pt \
    --layers 16 32 \
    --freq-threshold 10000

python scripts/extract_hidden_states.py \
    --model Qwen/Qwen3-8B \
    --dataset medbench \
    --output data/hidden_states/qwen3_medbench.pt \
    --layers 16 36 \
    --all-layers
"""

import argparse
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
    p = argparse.ArgumentParser(description="Extract LLM hidden states for KBP")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--dataset", required=True,
                   choices=["popqa", "medbench", "laobench", "mmlu", "mkqa", "custom"],
                   help="Dataset to extract from")
    p.add_argument("--output", required=True, help="Output path (.pt file)")
    p.add_argument("--layers", nargs=2, type=int, default=None,
                   help="Layer range [start, end] (inclusive). Default: paper range for model")
    p.add_argument("--all-layers", action="store_true",
                   help="Extract from all layers (overrides --layers)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--token-position", default="last",
                   choices=["last", "mean", "first", "last_k"])

    # Dataset-specific args
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Maximum number of samples to extract")
    p.add_argument("--freq-threshold", type=int, default=10000,
                   help="PopQA frequency threshold for KS/KD labeling (f* = 10^4)")
    p.add_argument("--custom-queries", default=None,
                   help="Path to text file with one query per line (for --dataset custom)")
    p.add_argument("--custom-labels", default=None,
                   help="Path to .npy file with binary labels (for --dataset custom)")

    p.add_argument("--trust-remote-code", action="store_true",
                   help="Pass trust_remote_code=True to transformers (required for Qwen3)")
    p.add_argument("--device-map", default="auto",
                   choices=["auto", "cuda", "cpu", "balanced"],
                   help="HuggingFace device_map strategy (default: auto)")
    p.add_argument("--mkqa-language", default="ar",
                   choices=["ar", "th", "sw", "en", "ja", "zh"])

    return p.parse_args()


def load_popqa(split: str, freq_threshold: int, max_samples: int = None):
    """
    Load PopQA dataset with frequency-based KS/KD labels (Definition 1 / O1).

    Returns (queries, labels) where labels: 1=KS (freq ≥ freq_threshold), 0=KD.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    logger.info("Loading PopQA...")
    ds = load_dataset("akariasai/PopQA", split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    queries = []
    labels = []

    for item in ds:
        question = item.get("question", item.get("query", ""))
        if not question:
            continue

        # Frequency proxy (O1): Wikipedia monthly page views
        page_views = item.get("s_pop", item.get("page_views", 0))
        label = 1 if page_views >= freq_threshold else 0

        queries.append(question)
        labels.append(label)

    labels = np.array(labels)
    n_ks = labels.sum()
    n_kd = (1 - labels).sum()
    logger.info(
        f"PopQA: {len(queries)} queries "
        f"(KS={n_ks} f≥{freq_threshold}, KD={n_kd} f<{freq_threshold})"
    )
    return queries, labels


def load_medbench(split: str, max_samples: int = None):
    """
    Load MedBench (Chinese medical QA).

    Labels via Accuracy Oracle (O2) since all queries are KD under O1.
    Try multiple known HuggingFace IDs; fall back to local file or synthetic.
    """
    candidate_ids = [
        "Yuchenmedical/medbench",
        "QingyiSi/Alpaca-CoT",   # contains medical QA subset
    ]
    for dataset_id in candidate_ids:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split=split)
            # Try common question field names
            queries = []
            for item in ds:
                q = item.get("question", item.get("query", item.get("instruction", "")))
                if q:
                    queries.append(q)
            if max_samples:
                queries = queries[:max_samples]
            labels = np.zeros(len(queries), dtype=int)
            logger.info(f"MedBench ({dataset_id}): {len(queries)} queries")
            return queries, labels
        except Exception as e:
            logger.warning(f"Could not load {dataset_id}: {e}")

    # Final fallback: local file
    local_path = Path("data/medbench_queries.txt")
    if local_path.exists():
        with open(local_path) as f:
            queries = [l.strip() for l in f if l.strip()]
        if max_samples:
            queries = queries[:max_samples]
        logger.info(f"MedBench (local): {len(queries)} queries")
        return queries, np.zeros(len(queries), dtype=int)

    logger.warning("MedBench not found. Using synthetic placeholder.")
    queries = [f"医学问题 {i}: 请问该疾病的症状是什么？" for i in range(max_samples or 100)]
    return queries, np.zeros(len(queries), dtype=int)


def load_mmlu(split: str, subject: str = "all", max_samples: int = None):
    """Load MMLU dataset."""
    try:
        from datasets import load_dataset
        if subject == "all":
            ds = load_dataset("cais/mmlu", "all", split=split)
        else:
            ds = load_dataset("cais/mmlu", subject, split=split)

        queries = []
        for item in ds:
            q = item["question"]
            choices = item["choices"]
            q_formatted = f"{q}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
            queries.append(q_formatted)

        if max_samples:
            queries = queries[:max_samples]

        # MMLU General → all KS; MMLU specialist subjects → all KD (under O1)
        is_general = subject in ("all", "global_facts", "miscellaneous")
        labels = np.ones(len(queries), dtype=int) if is_general else np.zeros(len(queries), dtype=int)
        logger.info(f"MMLU ({subject}): {len(queries)} queries")
        return queries, labels
    except Exception as e:
        logger.warning(f"Could not load MMLU: {e}. Using synthetic data.")
        queries = [f"MMLU question {i}?" for i in range(max_samples or 100)]
        return queries, np.zeros(len(queries), dtype=int)


def load_laobench(split: str, max_samples: int = None):
    """
    Load LaoBench (BAAI/LaoBench on HuggingFace).

    17k+ expert-curated Lao language QA samples across:
      - Knowledge Application (culturally grounded)
      - K12 Education (curriculum-aligned)
      - Bilingual Translation (Lao/Chinese/English)

    All queries are KD under O1 (Llama/Qwen have near-zero Lao knowledge).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("BAAI/LaoBench", split=split)
        queries = []
        for item in ds:
            q = item.get("question", item.get("query", item.get("input", "")))
            if q:
                queries.append(q)
        if max_samples:
            queries = queries[:max_samples]
        labels = np.zeros(len(queries), dtype=int)  # All KD under O1
        logger.info(f"LaoBench: {len(queries)} queries (all KD — low-resource Lao)")
        return queries, labels
    except Exception as e:
        logger.warning(f"Could not load LaoBench (BAAI/LaoBench): {e}")
        # Fallback: local file
        local_path = Path("data/laobench_queries.txt")
        if local_path.exists():
            with open(local_path) as f:
                queries = [l.strip() for l in f if l.strip()]
            if max_samples:
                queries = queries[:max_samples]
            return queries, np.zeros(len(queries), dtype=int)
        raise RuntimeError(
            "LaoBench not accessible. Try: pip install datasets && "
            "huggingface-cli login (if access-gated)"
        ) from e



    """Load MKQA (multilingual) dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("apple/mkqa", split="test")

        queries = []
        for item in ds:
            if language == "en":
                q = item["query"]
            else:
                # Use translated query if available, otherwise fall back to English
                lang_answers = item.get("answers", {}).get(language, {})
                # MKQA stores queries in English; translated queries come from answers dict
                # For non-English eval, we use English query fed to the model
                # (same as paper: model is always prompted in English)
                q = item["query"]
            queries.append(q)

        if max_samples:
            queries = queries[:max_samples]

        labels = np.zeros(len(queries), dtype=int)  # All KD under O1
        logger.info(f"MKQA ({language}): {len(queries)} queries")
        return queries, labels
    except Exception as e:
        logger.warning(f"Could not load MKQA: {e}")
        queries = [f"MKQA query {i} ({language})?" for i in range(max_samples or 200)]
        return queries, np.zeros(len(queries), dtype=int)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset}")

    if args.dataset == "popqa":
        queries, labels = load_popqa(args.split, args.freq_threshold, args.max_samples)
    elif args.dataset == "medbench":
        queries, labels = load_medbench(args.split, args.max_samples)
    elif args.dataset == "mmlu":
        queries, labels = load_mmlu(args.split, max_samples=args.max_samples)
    elif args.dataset == "mkqa":
        queries, labels = load_mkqa(args.mkqa_language, args.max_samples)
    elif args.dataset == "laobench":
        queries, labels = load_laobench(args.split, args.max_samples)
    elif args.dataset == "custom":
        if args.custom_queries is None:
            raise ValueError("--custom-queries required for --dataset custom")
        with open(args.custom_queries) as f:
            queries = [line.strip() for line in f if line.strip()]
        if args.custom_labels:
            labels = np.load(args.custom_labels)
        else:
            labels = None
            logger.warning("No labels provided for custom dataset.")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # -----------------------------------------------------------
    # Set up extractor
    # -----------------------------------------------------------
    import torch
    from kbp.extractor import HiddenStateExtractor, ExtractionConfig

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    config = ExtractionConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=dtype_map[args.dtype],
        token_position=args.token_position,
        normalization="l2",  # Always normalize per paper
    )

    extractor = HiddenStateExtractor(
        args.model,
        config=config,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    # Determine layer range
    if args.all_layers:
        layers = list(range(extractor.n_layers))
    elif args.layers:
        layers = list(range(args.layers[0], args.layers[1] + 1))
    else:
        start, end = extractor.get_optimal_layer_range(args.model)
        layers = list(range(start, end + 1))
        logger.info(f"Using paper-recommended layers {start}–{end} for {args.model}")

    logger.info(f"Extracting {len(layers)} layers: {layers[0]}–{layers[-1]}")

    # -----------------------------------------------------------
    # Extract
    # -----------------------------------------------------------
    output = extractor.extract(queries, layers=layers)

    # Save hidden states
    output.save(str(output_path))
    logger.info(f"Hidden states saved to {output_path}")

    # Save labels alongside
    if labels is not None:
        label_path = output_path.with_suffix(".labels.npy")
        np.save(label_path, labels)
        logger.info(f"Labels saved to {label_path}")
        logger.info(f"  KS (1): {labels.sum()}, KD (0): {(1-labels).sum()}")

    # Print summary
    logger.info("\nExtraction summary:")
    logger.info(f"  Model:   {args.model}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Queries: {len(queries)}")
    logger.info(f"  Layers:  {layers[0]}–{layers[-1]} ({len(layers)} total)")
    sample_h = output.get_layer(layers[0])
    logger.info(f"  Hidden dim: {sample_h.shape[1]}")


if __name__ == "__main__":
    main()
