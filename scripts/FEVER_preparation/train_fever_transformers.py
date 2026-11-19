#!/usr/bin/env python3
"""Fine-tune pretrained FEVER relevance and NLI models on prepared pair files.

This script does not touch the live OSINT database, MN-DS event classifier,
or existing pretrained model caches. Checkpoints and metrics are written to
a separate output directory.

Relevance: one-logit MS MARCO cross-encoder, trained with binary BCE loss.
Stance: pretrained 3-way NLI model, trained on gold FEVER evidence only.
The stance dataset has SUPPORTS/REFUTES only; it does not train a genuine
NEI/neutral class.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
)

GOLD = "fever_gold_evidence"
SYNTHETIC = "synthetic_random_negative"
DEFAULT_REL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_NLI = "cross-encoder/nli-deberta-v3-base"


class DataError(RuntimeError):
    pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pairs(path: Path, expected_split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataError(f"Missing pair file: {path}")
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            for key in ("pair_id", "claim_id", "claim_text", "passage_text",
                        "relevance_label", "stance_label", "split", "label_origin"):
                if key not in row:
                    raise DataError(f"{path}:{line_no}: missing {key}")
            if row["split"] != expected_split:
                raise DataError(f"{path}:{line_no}: expected split {expected_split!r}")
            if row["pair_id"] in seen:
                raise DataError(f"{path}:{line_no}: duplicate pair_id")
            seen.add(row["pair_id"])
            if row["label_origin"] == GOLD:
                if row["relevance_label"] != "relevant" or row["stance_label"] not in {"supports", "contradicts"}:
                    raise DataError(f"{path}:{line_no}: inconsistent gold labels")
            elif row["label_origin"] == SYNTHETIC:
                if row["relevance_label"] != "irrelevant" or row["stance_label"] is not None:
                    raise DataError(f"{path}:{line_no}: synthetic negatives must be irrelevant with null stance")
            else:
                raise DataError(f"{path}:{line_no}: unknown label_origin {row['label_origin']!r}")
            if not str(row["claim_text"]).strip() or not str(row["passage_text"]).strip():
                raise DataError(f"{path}:{line_no}: empty claim/passage")
            rows.append(row)
    if not rows:
        raise DataError(f"No rows in {path}")
    return rows


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(requested)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise DataError("MPS requested but unavailable")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise DataError("CUDA requested but unavailable")
    return dev


def encode_batch(tokenizer, rows, start, batch_size, max_length, task):
    batch = rows[start:start + batch_size]
    if task == "relevance":
        first = [r["claim_text"] for r in batch]
        second = [r["passage_text"] for r in batch]
    else:
        # NLI convention: premise=evidence passage, hypothesis=claim.
        first = [r["passage_text"] for r in batch]
        second = [r["claim_text"] for r in batch]
    return tokenizer(first, second, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt"), batch


def labels_for(rows, task, label_map=None):
    if task == "relevance":
        return torch.tensor([1.0 if r["relevance_label"] == "relevant" else 0.0 for r in rows])
    entail = next((i for i, s in label_map.items() if "entail" in s.lower()), None)
    contra = next((i for i, s in label_map.items() if "contrad" in s.lower()), None)
    if entail is None or contra is None:
        raise DataError(f"Cannot map NLI labels from model config: {label_map}")
    # Keep the pretrained 3-way head; FEVER supports -> entailment,
    # FEVER refutes -> contradiction. Neutral is not a training label here.
    return torch.tensor([entail if r["stance_label"] == "supports" else contra for r in rows],
                        dtype=torch.long)


def evaluate(model, tokenizer, rows, task, device, batch_size, max_length, label_map=None):
    model.eval()
    all_logits = []
    all_truth = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            enc, batch = encode_batch(tokenizer, rows, start, batch_size, max_length, task)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu()
            all_logits.append(logits)
            all_truth.extend(labels_for(batch, task, label_map).tolist())
    logits = torch.cat(all_logits).numpy()
    if task == "relevance":
        scores = logits.reshape(-1)
        truth = np.asarray(all_truth, dtype=np.int64)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
        pred = (probs >= 0.5).astype(int)
        return {
            "pair_count": len(rows),
            "labels": {"irrelevant": int((truth == 0).sum()), "relevant": int((truth == 1).sum())},
            "roc_auc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
            "accuracy_at_0_5": float(accuracy_score(truth, pred)),
            "macro_f1_at_0_5": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                truth, pred, labels=[0, 1], target_names=["irrelevant", "relevant"],
                output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
            "threshold": 0.5,
        }
    truth = np.asarray(all_truth, dtype=np.int64)
    pred = logits.argmax(axis=1)
    # Only entailment/contradiction are represented in the gold labels.
    entail = next(i for i, s in label_map.items() if "entail" in s.lower())
    contra = next(i for i, s in label_map.items() if "contrad" in s.lower())
    id_to_fever = {entail: "supports", contra: "contradicts"}
    y_true = ["supports" if x == entail else "contradicts" for x in truth]
    y_pred = [id_to_fever.get(int(x), "neutral") for x in pred]
    labels = ["contradicts", "supports", "neutral"]
    return {
        "pair_count": len(rows),
        "gold_label_counts": dict(Counter(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support_refute_macro_f1": float(f1_score(y_true, y_pred, labels=["contradicts", "supports"],
                                                   average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                         output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "label_order": labels,
        "neutral_prediction_rate": float(sum(x == "neutral" for x in y_pred) / len(y_pred)),
        "note": "No genuine NEI/neutral training examples are present; neutral predictions count as errors.",
    }


def train_one(task, train_rows, val_rows, args, device):
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    model_name = args.relevance_model if task == "relevance" else args.nli_model
    print(f"\n[{task}] Loading {model_name}", flush=True)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    # Reduce activation memory for DeBERTa on Apple MPS.
    if device.type == "mps" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    label_map = {int(k): str(v) for k, v in config.id2label.items()}
    if task == "relevance" and int(config.num_labels) != 1:
        raise DataError(f"Expected one-logit MS MARCO model, got num_labels={config.num_labels}")
    if task == "stance":
        if not any("entail" in v.lower() for v in label_map.values()) or not any("contrad" in v.lower() for v in label_map.values()):
            raise DataError(f"Unexpected NLI label mapping: {label_map}")
        train_rows = [r for r in train_rows if r["label_origin"] == GOLD and r["stance_label"] in {"supports", "contradicts"}]
        val_rows = [r for r in val_rows if r["label_origin"] == GOLD and r["stance_label"] in {"supports", "contradicts"}]
    if not train_rows or not val_rows:
        raise DataError(f"No training or validation rows for {task}")

    # AdamW; small LR is important when adapting pretrained cross-encoders.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    best_metric = -float("inf")
    task_dir = Path(args.output_dir) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        steps = 0
        start_time = time.time()
        for batch_no, offset in enumerate(range(0, len(order), args.batch_size), 1):
            idx = order[offset:offset + args.batch_size]
            batch = [train_rows[i] for i in idx]
            # Tokenize directly in batches to bound memory.
            if task == "relevance":
                first = [r["claim_text"] for r in batch]
                second = [r["passage_text"] for r in batch]
            else:
                first = [r["passage_text"] for r in batch]
                second = [r["claim_text"] for r in batch]
            enc = tokenizer(first, second, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if task == "relevance":
                y = torch.tensor([1.0 if r["relevance_label"] == "relevant" else 0.0 for r in batch],
                                 dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), y)
            else:
                y = labels_for(batch, task, label_map).to(device)
                loss = torch.nn.functional.cross_entropy(logits, y)
            (loss / args.grad_accum).backward()
            if batch_no % args.grad_accum == 0 or offset + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            running_loss += float(loss.detach().cpu())
            if batch_no % 500 == 0:
                print(f"[{task}] epoch {epoch}/{args.epochs}: {batch_no:,}/{math.ceil(len(order)/args.batch_size):,} batches",
                      flush=True)
        elapsed = time.time() - start_time
        metrics = evaluate(model, tokenizer, val_rows, task, device,
                           args.eval_batch_size, args.max_length, label_map)
        score = metrics["roc_auc"] if task == "relevance" else metrics["support_refute_macro_f1"]
        record = {"epoch": epoch, "mean_training_loss": running_loss / max(1, math.ceil(len(order)/args.batch_size)),
                  "validation": metrics, "epoch_seconds": elapsed}
        history.append(record)
        print(f"[{task}] epoch {epoch}: loss={record['mean_training_loss']:.4f}, "
              f"validation_score={score:.4f}, time={elapsed:.1f}s", flush=True)
        if score > best_metric:
            best_metric = score
            model.save_pretrained(task_dir / "best")
            tokenizer.save_pretrained(task_dir / "best")
            (task_dir / "best_metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (task_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if device.type == "mps":
            torch.mps.empty_cache()

    result = {"task": task, "model_name": model_name, "best_validation_score": best_metric,
              "checkpoint": str(task_dir / "best"), "epochs_completed": args.epochs}
    # Release model and optimizer memory before another task is loaded.
    del model, optimizer, scheduler, tokenizer, config
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
        gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/fever/processed_pairs"))
    p.add_argument("--output-dir", type=Path, default=Path("models/fever_finetuned"))
    p.add_argument("--task", choices=["all", "relevance", "stance"], default="all")
    p.add_argument("--relevance-model", default=DEFAULT_REL)
    p.add_argument("--nli-model", default=DEFAULT_NLI)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.grad_accum < 1:
        p.error("epochs, batch sizes and grad-accum must be >= 1")
    if args.max_length < 8 or args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        p.error("invalid max-length, learning-rate or warmup-ratio")
    try:
        seed_everything(args.seed)
        device = device_for(args.device)
        train_rows = read_pairs(args.data_dir / "train_pairs.jsonl", "train")
        val_rows = read_pairs(args.data_dir / "validation_pairs.jsonl", "validation")
        overlap = {r["claim_id"] for r in train_rows} & {r["claim_id"] for r in val_rows}
        if overlap:
            raise DataError(f"Claim leakage across train/validation; example {next(iter(overlap))}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}; train pairs: {len(train_rows):,}; validation pairs: {len(val_rows):,}", flush=True)
        tasks = ["relevance", "stance"] if args.task == "all" else [args.task]
        results = []
        for task in tasks:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            results.append(train_one(task, train_rows, val_rows, args, device))
        summary = {"dataset": "FEVER v1.0 prepared pairs, preparation_version=2",
                   "pretrained_fine_tuning": True, "device": str(device),
                   "training_rows": len(train_rows), "validation_rows": len(val_rows),
                   "results": results,
                   "limitations": [
                       "Relevance includes synthetic random negatives; validation metrics do not measure hard-negative performance.",
                       "Stance is trained only on gold SUPPORTS/REFUTES evidence; no genuine NEI/neutral training class.",
                       "The FEVER development split is used for epoch/model selection, not an unbiased final test.",
                       "These checkpoints are separate from the live OSINT pipeline and MN-DS event classifier."
                   ]}
        (args.output_dir / "finetuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("\nFINE-TUNING COMPLETE")
        print(f"Checkpoints and metrics: {args.output_dir.resolve()}")
        for result in results:
            print(f"{result['task']}: {result['checkpoint']} (best validation score={result['best_validation_score']:.4f})")
        return 0
    except (OSError, DataError, ValueError, RuntimeError) as exc:
        print(f"FINE-TUNING FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
