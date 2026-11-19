#!/usr/bin/env python3
"""Evaluate fine-tuned FEVER relevance and NLI checkpoints on prepared pairs.

Uses validation_pairs.jsonl; no Wikipedia archive is needed. Loads local
fine-tuned checkpoints and never modifies checkpoints or the live OSINT database.

Relevance model: cross-encoder/ms-marco-MiniLM-L6-v2
  Outputs a ranking score, not a calibrated probability. Report ROC-AUC and
  average precision; thresholded metrics at score >= 0 are diagnostic only.

NLI model: cross-encoder/nli-deberta-v3-base
  Input order: premise=passage, hypothesis=claim. Maps FEVER SUPPORTS to
  entailment and REFUTES to contradiction. Evaluates only gold evidence pairs.

Synthetic negatives are weak/easy relevance negatives. The script reports their
metrics separately and warns that this is not a realistic hard-negative test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)


DEFAULT_RELEVANCE_MODEL = "models/fever_finetuned/relevance/best"
DEFAULT_NLI_MODEL = "models/fever_finetuned/stance/best"
GOLD_ORIGIN = "fever_gold_evidence"
SYNTHETIC_ORIGIN = "synthetic_random_negative"


class EvaluationError(RuntimeError):
    pass


def read_jsonl(path: Path, split: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"Missing {split} file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise EvaluationError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            for field in ("claim_text", "passage_text", "relevance_label", "label_origin"):
                if field not in row:
                    raise EvaluationError(f"{path}:{line_no}: missing {field}")
            if row["relevance_label"] not in {"relevant", "irrelevant"}:
                raise EvaluationError(f"{path}:{line_no}: invalid relevance_label")
            if row["label_origin"] not in {GOLD_ORIGIN, SYNTHETIC_ORIGIN}:
                raise EvaluationError(f"{path}:{line_no}: unknown label_origin {row['label_origin']!r}")
            rows.append(row)
    if not rows:
        raise EvaluationError(f"No rows in {path}")
    return rows


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise EvaluationError("MPS requested but PyTorch MPS is unavailable.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA requested but CUDA is unavailable.")
    return torch.device(requested)


def load_model(model_name: str, device: torch.device):
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise EvaluationError(
            "Missing dependency 'transformers'. Install it in .venv with: "
            "python -m pip install transformers sentencepiece"
        ) from exc
    # Explicitly load the model's own label configuration; do not assume class IDs.
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return config, tokenizer, model


def infer_logits(
    tokenizer, model, device: torch.device,
    text_pairs: Sequence[Tuple[str, str]], batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_logits = []
    with torch.inference_mode():
        for start in range(0, len(text_pairs), batch_size):
            batch = text_pairs[start:start + batch_size]
            first = [x[0] for x in batch]
            second = [x[1] for x in batch]
            encoded = tokenizer(
                first, second, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            all_logits.append(logits.detach().float().cpu().numpy())
            done = min(start + len(batch), len(text_pairs))
            if done == len(text_pairs) or done % (batch_size * 20) == 0:
                print(f"  Scored {done:,}/{len(text_pairs):,}", flush=True)
    return np.concatenate(all_logits, axis=0)


def label_map(config) -> Dict[str, int]:
    result = {}
    for idx, label in config.id2label.items():
        normalized = str(label).lower().replace("_", "").replace("-", "").replace(" ", "")
        result[normalized] = int(idx)
    return result


def find_label(mapping: Dict[str, int], candidates: Sequence[str]) -> int:
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace("-", "").replace(" ", "")
        if key in mapping:
            return mapping[key]
    raise EvaluationError(
        f"Could not map expected label {candidates}; model labels are {mapping}"
    )


def relevance_metrics(rows: Sequence[Dict[str, Any]], scores: np.ndarray) -> Dict[str, Any]:
    y = np.array([1 if r["relevance_label"] == "relevant" else 0 for r in rows], dtype=np.int8)
    if len(np.unique(y)) != 2:
        raise EvaluationError("Relevance validation needs both relevant and irrelevant labels.")
    # MS MARCO is a ranker. Its raw score is not a calibrated probability.
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    threshold = 0.0  # diagnostic only; not claimed to be a calibrated decision boundary
    pred = (scores >= threshold).astype(np.int8)

    def subset(indices: List[int]) -> Dict[str, Any]:
        if not indices:
            return {}
        yy, ss = y[indices], scores[indices]
        pp = (ss >= threshold).astype(np.int8)
        return {
            "count": len(indices),
            "score_mean": float(np.mean(ss)),
            "score_median": float(np.median(ss)),
            "score_min": float(np.min(ss)),
            "score_max": float(np.max(ss)),
            "threshold_0_diagnostic": {
                "accuracy": float(accuracy_score(yy, pp)),
                "precision_recall_f1": {
                    "precision": float(precision_recall_fscore_support(
                        yy, pp, average="binary", zero_division=0
                    )[0]),
                    "recall": float(precision_recall_fscore_support(
                        yy, pp, average="binary", zero_division=0
                    )[1]),
                    "f1": float(f1_score(yy, pp, zero_division=0)),
                },
            },
        }

    gold_idx = [i for i, r in enumerate(rows) if r["label_origin"] == GOLD_ORIGIN]
    neg_idx = [i for i, r in enumerate(rows) if r["label_origin"] == SYNTHETIC_ORIGIN]
    return {
        "model_task": "MS MARCO passage ranking score used as a relevance signal",
        "pair_count": len(rows),
        "labels": {"relevant": int(y.sum()), "irrelevant": int((1-y).sum())},
        "ranking_metrics": {"roc_auc": auc, "average_precision": ap},
        "threshold_0_is_diagnostic_only": {
            "accuracy": float(accuracy_score(y, pred)),
            "classification_report": classification_report(
                y, pred, labels=[0, 1], target_names=["irrelevant", "relevant"],
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "label_order": ["irrelevant", "relevant"],
        },
        "gold_evidence_score_diagnostics": subset(gold_idx),
        "synthetic_negative_score_diagnostics": subset(neg_idx),
        "warning": (
            "The MS MARCO model returns a ranking score, not a calibrated probability. "
            "Threshold 0 is diagnostic only. Synthetic negatives are easy weak labels; "
            "these metrics do not establish performance on hard retrieved news passages."
        ),
    }


def stance_metrics(rows: Sequence[Dict[str, Any]], logits: np.ndarray, config) -> Dict[str, Any]:
    selected = [
        (i, r) for i, r in enumerate(rows)
        if r["label_origin"] == GOLD_ORIGIN and r.get("stance_label") in {"supports", "contradicts"}
    ]
    if not selected:
        raise EvaluationError("No gold SUPPORTS/REFUTES rows available for NLI evaluation.")
    mapping = label_map(config)
    entail_idx = find_label(mapping, ["entailment", "entails"])
    contra_idx = find_label(mapping, ["contradiction", "contradict"])
    neutral_idx = find_label(mapping, ["neutral"])
    # `logits` is passed in the same filtered gold-evidence row order.
    if len(logits) != len(selected):
        raise EvaluationError(
            f"NLI logits/rows mismatch: {len(logits)} logits for {len(selected)} gold pairs"
        )
    selected_logits = logits
    probs = torch.softmax(torch.tensor(selected_logits), dim=1).numpy()
    predicted_nli_idx = probs.argmax(axis=1)
    # Evaluate the binary FEVER task using only entailment vs contradiction.
    # If NLI predicts neutral, it is counted as incorrect for either gold class.
    pred = []
    truth = []
    for j, (_, row) in enumerate(selected):
        if predicted_nli_idx[j] == entail_idx:
            pred.append("supports")
        elif predicted_nli_idx[j] == contra_idx:
            pred.append("contradicts")
        else:
            pred.append("neutral")
        truth.append(row["stance_label"])
    labels = ["contradicts", "supports", "neutral"]
    cm_labels = ["contradicts", "supports", "neutral"]
    report = classification_report(
        truth, pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "model_task": "3-way NLI evaluated against FEVER gold SUPPORTS/REFUTES evidence",
        "pair_count": len(selected),
        "gold_label_counts": dict(Counter(truth)),
        "nli_model_label_mapping": {str(k): v for k, v in config.id2label.items()},
        "nli_label_indices": {
            "entailment": entail_idx, "contradiction": contra_idx, "neutral": neutral_idx
        },
        "metrics": {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1_over_support_refute_neutral": float(
                f1_score(truth, pred, labels=labels, average="macro", zero_division=0)
            ),
            "support_refute_only_macro_f1": float(
                f1_score(truth, pred, labels=["contradicts", "supports"],
                         average="macro", zero_division=0)
            ),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(truth, pred, labels=cm_labels).tolist(),
            "label_order": cm_labels,
            "neutral_prediction_rate": float(sum(p == "neutral" for p in pred) / len(pred)),
        },
        "warning": (
            "FEVER stance evaluation uses gold evidence only and only SUPPORTS/REFUTES claims. "
            "Neutral predictions count as errors here; this does not evaluate genuine NOT ENOUGH INFO "
            "claims or retrieved-passage verification."
        ),
    }


def evaluate(args) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    rows = read_jsonl(data_dir / "validation_pairs.jsonl", "validation")
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relevance_result = None
    stance_result = None
    timings = {}

    if args.task in {"all", "relevance"}:
        print(f"Loading relevance model: {args.relevance_model}", flush=True)
        config, tokenizer, model = load_model(args.relevance_model, device)
        # Inputs are query=claim, document=passage, as expected by MS MARCO cross-encoder.
        pairs = [(r["claim_text"], r["passage_text"]) for r in rows]
        start = time.time()
        logits = infer_logits(tokenizer, model, device, pairs, args.batch_size, args.max_length)
        timings["relevance_seconds"] = time.time() - start
        if logits.shape[1] != 1:
            # Some repository revisions expose two logits. Use positive-class label if named.
            if logits.shape[1] == 2:
                mapping = label_map(config)
                positive_idx = find_label(mapping, ["relevant", "positive", "1"])
                scores = logits[:, positive_idx]
            else:
                raise EvaluationError(f"Expected one relevance score, got logits shape {logits.shape}")
        else:
            scores = logits[:, 0]
        relevance_result = relevance_metrics(rows, scores)
        del model, tokenizer, config, logits
        import gc
        del pairs, scores
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    if args.task in {"all", "stance"}:
        # Stance model is only scored on gold evidence pairs, reducing compute.
        stance_rows = [
            r for r in rows
            if r["label_origin"] == GOLD_ORIGIN and r.get("stance_label") in {"supports", "contradicts"}
        ]
        print(f"Loading NLI model: {args.nli_model}", flush=True)
        config, tokenizer, model = load_model(args.nli_model, device)
        # NLI convention: premise=evidence passage; hypothesis=claim.
        pairs = [(r["passage_text"], r["claim_text"]) for r in stance_rows]
        start = time.time()
        logits = infer_logits(tokenizer, model, device, pairs, args.batch_size, args.max_length)
        timings["stance_seconds"] = time.time() - start
        stance_result = stance_metrics(stance_rows, logits, config)
        del model, tokenizer, config, logits
        import gc
        del pairs
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()

    result = {
        "dataset": "FEVER v1.0 prepared pairs; validation split only",
        "pretrained_only": False,
        "fine_tuned": True,
        "checkpoint_evaluation": True,
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("importlib.metadata", fromlist=["version"]).version("transformers"),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "models": {
            "relevance": args.relevance_model if relevance_result is not None else None,
            "stance": args.nli_model if stance_result is not None else None,
        },
        "validation_pair_count": len(rows),
        "timings": timings,
        "relevance": relevance_result,
        "stance": stance_result,
        "interpretation": [
            "No Wikipedia pages are read; the script uses prepared claim-passage pairs only.",
            "Local fine-tuned checkpoints are loaded; no training occurs during evaluation.",
            "MS MARCO relevance output is a ranking score, not a calibrated probability.",
            "FEVER synthetic negatives are easy weak negatives; not a hard-negative benchmark.",
            "Stance evaluation is on gold evidence for SUPPORTS/REFUTES only.",
            "Validation metrics are model-selection results, not an unbiased final test estimate.",
        ],
    }
    out = output_dir / "finetuned_validation_metrics.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(f"\nEvaluation complete: {out}", flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/fever/processed_pairs")
    p.add_argument("--output-dir", default="reports/fever_finetuned_evaluation")
    p.add_argument("--relevance-model", default=DEFAULT_RELEVANCE_MODEL)
    p.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    p.add_argument("--task", choices=["all", "relevance", "stance"], default="all")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    if args.batch_size < 1 or args.max_length < 8:
        p.error("--batch-size must be >=1 and --max-length must be >=8")
    for task, model_path in (("relevance", args.relevance_model), ("stance", args.nli_model)):
        if args.task in {"all", task} and not Path(model_path).exists():
            p.error(f"Fine-tuned {task} checkpoint does not exist: {model_path}")
    try:
        evaluate(args)
        return 0
    except (EvaluationError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
