#!/usr/bin/env python3
"""Train separate FEVER relevance and stance baselines.

This is a conservative CPU-friendly baseline for the prepared FEVER v1.0 pairs.
It does not modify the live OSINT database or existing models.

Relevance model:
  - trains on gold FEVER evidence (relevant) and marked synthetic negatives
    (irrelevant). Synthetic-negative metrics are weak-label diagnostics only.

Stance model:
  - trains ONLY on gold FEVER evidence pairs with supports/contradicts labels.
  - does not invent or train an insufficient class from random unrelated text.

The two tasks share one train-fitted TF-IDF vectorizer, then use separate
LinearSVC classifiers. Validation data is never used to fit vocabulary or IDF.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.svm import LinearSVC
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit(
        "Missing training dependency. Activate OSINT_final/.venv and ensure "
        "scikit-learn, numpy and joblib are installed. Original error: " + str(exc)
    ) from exc

REQUIRED_FIELDS = {
    "pair_id", "claim_id", "claim_text", "passage_text", "relevance_label",
    "stance_label", "claim_level_label", "split", "label_origin",
    "is_synthetic_label",
}
VALID_RELEVANCE = {"relevant", "irrelevant"}
VALID_STANCE = {"supports", "contradicts"}
GOLD_ORIGIN = "fever_gold_evidence"
SYNTHETIC_ORIGIN = "synthetic_random_negative"


class TrainingDataError(ValueError):
    """Raised when prepared pair data is malformed or inconsistent."""


def read_jsonl(path: Path, expected_split: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise TrainingDataError(f"Required input file not found: {path}")
    rows: List[Dict[str, Any]] = []
    pair_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise TrainingDataError(
                    f"{path}:{line_number}: missing fields {sorted(missing)}"
                )
            if row["split"] != expected_split:
                raise TrainingDataError(
                    f"{path}:{line_number}: expected split {expected_split!r}, got {row['split']!r}"
                )
            if not str(row["claim_text"]).strip() or not str(row["passage_text"]).strip():
                raise TrainingDataError(f"{path}:{line_number}: empty claim or passage")
            if row["relevance_label"] not in VALID_RELEVANCE:
                raise TrainingDataError(f"{path}:{line_number}: invalid relevance label")
            if row["pair_id"] in pair_ids:
                raise TrainingDataError(f"{path}:{line_number}: duplicate pair_id {row['pair_id']!r}")
            pair_ids.add(row["pair_id"])

            origin = row["label_origin"]
            stance = row["stance_label"]
            if origin == GOLD_ORIGIN:
                if row["is_synthetic_label"] is not False:
                    raise TrainingDataError(f"{path}:{line_number}: gold row marked synthetic")
                if row["relevance_label"] != "relevant" or stance not in VALID_STANCE:
                    raise TrainingDataError(f"{path}:{line_number}: invalid gold pair labels")
            elif origin == SYNTHETIC_ORIGIN:
                if row["is_synthetic_label"] is not True:
                    raise TrainingDataError(f"{path}:{line_number}: synthetic row not marked synthetic")
                if row["relevance_label"] != "irrelevant" or stance is not None:
                    raise TrainingDataError(
                        f"{path}:{line_number}: synthetic negatives must be irrelevant with null stance"
                    )
            else:
                raise TrainingDataError(f"{path}:{line_number}: unknown label origin {origin!r}")
            rows.append(row)
    if not rows:
        raise TrainingDataError(f"No records found in {path}")
    return rows


def pair_text(row: Dict[str, Any]) -> str:
    # Explicit separator helps the vectorizer distinguish claim from passage.
    return f"{row['claim_text']} [PASSAGE] {row['passage_text']}"


def safe_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "label_order": list(labels),
    }


def evaluate_relevance(rows: Sequence[Dict[str, Any]], y_pred: Sequence[str]) -> Dict[str, Any]:
    y_true = [row["relevance_label"] for row in rows]
    report: Dict[str, Any] = {
        "all_validation_pairs": safe_report(y_true, y_pred, ["irrelevant", "relevant"]),
        "gold_evidence_only": {},
        "synthetic_negative_only": {},
        "warning": (
            "Synthetic negatives are easy weak labels selected by zero content-token overlap. "
            "Their scores are not a reliable estimate of real-world irrelevant-passage detection."
        ),
    }
    gold_indices = [i for i, row in enumerate(rows) if row["label_origin"] == GOLD_ORIGIN]
    neg_indices = [i for i, row in enumerate(rows) if row["label_origin"] == SYNTHETIC_ORIGIN]
    if gold_indices:
        gold_true = [rows[i]["relevance_label"] for i in gold_indices]
        gold_pred = [y_pred[i] for i in gold_indices]
        report["gold_evidence_only"] = {
            "gold_positive_recall": float(recall_score(gold_true, gold_pred, pos_label="relevant", zero_division=0)),
            "gold_positive_precision": float(precision_score(gold_true, gold_pred, pos_label="relevant", zero_division=0)),
            "gold_pair_count": len(gold_indices),
        }
    if neg_indices:
        neg_true = [rows[i]["relevance_label"] for i in neg_indices]
        neg_pred = [y_pred[i] for i in neg_indices]
        report["synthetic_negative_only"] = {
            "synthetic_negative_accuracy": float(accuracy_score(neg_true, neg_pred)),
            "synthetic_negative_false_relevant_rate": float(sum(p == "relevant" for p in neg_pred) / len(neg_pred)),
            "synthetic_pair_count": len(neg_indices),
        }
    return report


def train(data_dir: Path, output_dir: Path, max_features: int, min_df: int, c_value: float) -> Dict[str, Any]:
    train_rows = read_jsonl(data_dir / "train_pairs.jsonl", "train")
    validation_rows = read_jsonl(data_dir / "validation_pairs.jsonl", "validation")

    train_claim_ids = {row["claim_id"] for row in train_rows}
    validation_claim_ids = {row["claim_id"] for row in validation_rows}
    overlap = train_claim_ids & validation_claim_ids
    if overlap:
        example = sorted(overlap)[0]
        raise TrainingDataError(
            f"Claim leakage between train and validation pair files; example claim_id={example!r}"
        )

    train_relevance_counts = Counter(row["relevance_label"] for row in train_rows)
    train_stance_rows = [
        row for row in train_rows
        if row["label_origin"] == GOLD_ORIGIN and row["stance_label"] in VALID_STANCE
    ]
    validation_stance_rows = [
        row for row in validation_rows
        if row["label_origin"] == GOLD_ORIGIN and row["stance_label"] in VALID_STANCE
    ]
    train_stance_counts = Counter(row["stance_label"] for row in train_stance_rows)
    validation_stance_counts = Counter(row["stance_label"] for row in validation_stance_rows)

    if len(train_relevance_counts) != 2 or set(train_relevance_counts) != VALID_RELEVANCE:
        raise TrainingDataError(f"Relevance training set must contain both classes: {train_relevance_counts}")
    if set(train_stance_counts) != VALID_STANCE:
        raise TrainingDataError(f"Gold stance training set must contain supports and contradicts: {train_stance_counts}")
    if set(validation_stance_counts) != VALID_STANCE:
        raise TrainingDataError(
            f"Gold stance validation set must contain supports and contradicts: {validation_stance_counts}"
        )

    # Fit vocabulary and IDF on train split only. Validation is transform-only.
    train_texts = [pair_text(row) for row in train_rows]
    validation_texts = [pair_text(row) for row in validation_rows]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    print(f"Vectorizing {len(train_texts):,} training pairs; validation pairs are transform-only.", flush=True)
    x_train = vectorizer.fit_transform(train_texts)
    x_validation = vectorizer.transform(validation_texts)
    del train_texts, validation_texts

    relevance_y = [row["relevance_label"] for row in train_rows]
    relevance_model = LinearSVC(C=c_value, class_weight="balanced", random_state=42, max_iter=5000)
    print("Training relevance classifier...", flush=True)
    relevance_model.fit(x_train, relevance_y)
    relevance_predictions = relevance_model.predict(x_validation)
    relevance_metrics = evaluate_relevance(validation_rows, relevance_predictions)

    stance_train_indices = [i for i, row in enumerate(train_rows) if row["label_origin"] == GOLD_ORIGIN]
    stance_validation_indices = [i for i, row in enumerate(validation_rows) if row["label_origin"] == GOLD_ORIGIN]
    stance_x_train = x_train[stance_train_indices]
    stance_x_validation = x_validation[stance_validation_indices]
    stance_y_train = [train_rows[i]["stance_label"] for i in stance_train_indices]
    stance_y_validation = [validation_rows[i]["stance_label"] for i in stance_validation_indices]
    stance_model = LinearSVC(C=c_value, class_weight="balanced", random_state=42, max_iter=5000)
    print(f"Training stance classifier on {len(stance_y_train):,} gold evidence pairs only...", flush=True)
    stance_model.fit(stance_x_train, stance_y_train)
    stance_predictions = stance_model.predict(stance_x_validation)
    stance_metrics = safe_report(stance_y_validation, stance_predictions, ["contradicts", "supports"])

    output_dir.mkdir(parents=True, exist_ok=True)
    # Write artifacts atomically to avoid leaving partial model files on failure.
    artifacts = {
        "tfidf_vectorizer.joblib": vectorizer,
        "relevance_linear_svc.joblib": relevance_model,
        "stance_linear_svc.joblib": stance_model,
    }
    for filename, artifact in artifacts.items():
        temp_path = output_dir / (filename + ".tmp")
        joblib.dump(artifact, temp_path, compress=3)
        os.replace(temp_path, output_dir / filename)

    metrics = {
        "dataset": "FEVER v1.0 prepared pairs, preparation_version=2",
        "training_method": "TF-IDF word unigrams+bigrams + LinearSVC",
        "input_directory": str(data_dir),
        "output_directory": str(output_dir),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": min_df,
            "max_features": max_features,
            "sublinear_tf": True,
            "dtype": "float32",
            "learned_feature_count": len(vectorizer.vocabulary_),
        },
        "classifier": {"type": "LinearSVC", "C": c_value, "class_weight": "balanced", "max_iter": 5000},
        "split_counts": {
            "train_pairs": len(train_rows),
            "validation_pairs": len(validation_rows),
            "train_relevance_labels": dict(train_relevance_counts),
            "train_gold_stance_labels": dict(train_stance_counts),
            "validation_gold_stance_labels": dict(validation_stance_counts),
            "claim_id_overlap": len(overlap),
        },
        "relevance_validation": relevance_metrics,
        "stance_validation_on_gold_evidence_only": stance_metrics,
        "interpretation_limits": [
            "Relevance validation negatives are synthetic random passages selected for zero shared content-token overlap; they are weak/easy negatives.",
            "Gold FEVER evidence is used as the relevant class; this evaluation does not measure ranking quality over a realistic candidate-retrieval pool.",
            "Stance is evaluated only on gold evidence for SUPPORTS/REFUTES claims; this does not measure NOT ENOUGH INFO or real-world retrieval-conditioned verification.",
            "These are classical ML baselines, not fine-tuned transformer models and not production factual-verification accuracy.",
        ],
        "artifacts": list(artifacts.keys()),
    }
    metrics_path = output_dir / "validation_metrics.json"
    metrics_tmp = output_dir / "validation_metrics.json.tmp"
    metrics_tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(metrics_tmp, metrics_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fever/processed_pairs"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fever_baselines"))
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_features < 1000 or args.min_df < 1 or args.c <= 0:
        parser.error("--max-features must be >= 1000, --min-df >= 1, and --c > 0")
    try:
        metrics = train(args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve(),
                        args.max_features, args.min_df, args.c)
    except (OSError, TrainingDataError, ValueError) as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nTRAINING COMPLETE")
    print(f"Models and metrics: {args.output_dir}")
    rel = metrics["relevance_validation"]["all_validation_pairs"]
    st = metrics["stance_validation_on_gold_evidence_only"]
    print(f"Relevance validation: accuracy={rel['accuracy']:.4f}, macro-F1={rel['macro_f1']:.4f}")
    print(f"Stance (gold evidence only): accuracy={st['accuracy']:.4f}, macro-F1={st['macro_f1']:.4f}")
    print("Caution: relevance negatives are synthetic/easy; stance has no NOT ENOUGH INFO class.")
    print("These are offline baselines; the live OSINT pipeline has not been modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
