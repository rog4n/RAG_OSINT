#!/usr/bin/env python3
"""Claim--evidence NLI scoring. Predictions are triage signals, not fact verdicts.

Reads current claim_evidence links from the project's SQLite database, scores each
(claim, passage) pair with a three-way NLI model, and writes predictions to a
separate table/CSV. Human assessment and retrieval records are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "new_articles.db"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_STANCE_CHECKPOINT = ROOT / "models" / "fever_finetuned" / "stance" / "best"
DEFAULT_RELEVANCE_CHECKPOINT = ROOT / "models" / "fever_finetuned" / "relevance" / "best"
DEFAULT_MODEL = str(DEFAULT_STANCE_CHECKPOINT) if DEFAULT_STANCE_CHECKPOINT.is_dir() else "cross-encoder/nli-roberta-base"
DEFAULT_RELEVANCE_MODEL = str(DEFAULT_RELEVANCE_CHECKPOINT) if DEFAULT_RELEVANCE_CHECKPOINT.is_dir() else ""
BATCH_SIZE = 16
MAX_LENGTH = 512

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("nli_verification")


@dataclass(frozen=True)
class NLIThresholds:
    min_confidence: float = 0.70
    min_margin: float = 0.20
    min_entailment: float = 0.70
    min_contradiction: float = 0.70


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_pair_id(claim_id: str, passage_id: str, model_name: str) -> str:
    raw = "\x1f".join((claim_id, passage_id, model_name)).encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:32]


def validate_thresholds(t: NLIThresholds) -> None:
    for name, value in vars(t).items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


# Conservative lexical/entity gate. It is a triage heuristic, not a factuality test.
_GATE_STOPWORDS = {
    "about","after","again","against","also","among","because","before","being",
    "between","both","could","during","each","from","have","having","into","more",
    "most","over","said","says","some","such","than","that","their","there",
    "these","they","this","those","through","under","very","were","what","when",
    "where","which","while","with","would","will","been","were","was","are","the",
    "and","for","but","not","you","your","his","her","its","our","out","who",
    "has","had","did","does","may","can","one","two","three","per","new","year",
    "years","day","days","report","reports","according","officials","official",
    "government","police","minister","president","country","state","city"
}
_GENERIC_ENTITY_WORDS = {
    "the","government","police","minister","president","official","officials",
    "country","state","city","court","company","people","authorities","report"
}

def _content_tokens(text: str) -> set[str]:
    # Unicode-aware tokenization preserves names such as Orbán and Péter.
    words = re.findall(r"[^\W\d_][\w'’-]{2,}", clean_text(text).lower(), flags=re.UNICODE)
    return {w.strip("'-") for w in words if w not in _GATE_STOPWORDS and len(w) >= 3}

def _proper_name_phrases(text: str) -> set[str]:
    # Capitalized multiword names and acronyms; intentionally conservative.
    matches = re.findall(
        r"\b(?:[A-ZÀ-ÖØ-Þ][\w'’-]{2,}|[A-Z]{2,})(?:\s+(?:[A-ZÀ-ÖØ-Þ][\w'’-]{2,}|[A-Z]{2,})){0,3}\b",
        clean_text(text)
    )
    result = set()
    for item in matches:
        norm = re.sub(r"[\W_]", "", item.lower(), flags=re.UNICODE).strip()
        words = norm.split()
        if not words or all(w in _GENERIC_ENTITY_WORDS for w in words):
            continue
        if words[0] in {"the","a","an"}:
            words = words[1:]
        if words and not all(w in _GENERIC_ENTITY_WORDS for w in words):
            result.add(" ".join(words))
    return result

def relevance_gate(claim: str, evidence: str, stored_shared_entities: str = "") -> dict:
    """Compute independent lexical/entity signals; this is not a factuality test."""
    c_tokens = _content_tokens(claim)
    e_tokens = _content_tokens(evidence)
    shared = c_tokens & e_tokens
    denom = max(1, min(len(c_tokens), len(e_tokens)))
    overlap_ratio = len(shared) / denom

    # These are conservative capitalization-derived entity candidates, not full NER.
    c_names = _proper_name_phrases(claim)
    e_names = _proper_name_phrases(evidence)
    name_overlap = c_names & e_names
    stored = {
        x.strip().lower() for x in re.split(r"[;,|]", clean_text(stored_shared_entities))
        if x.strip()
    }
    # Retrieval-stage shared entities are additional evidence of an entity match.
    name_overlap |= stored
    entity_union = c_names | e_names
    # Jaccard uses independently extracted text entities only; stored shared
    # entities remain a corroborating match signal but cannot inflate the ratio.
    entity_jaccard = len(c_names & e_names) / max(1, len(entity_union))

    c_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", clean_text(claim)))
    e_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", clean_text(evidence)))
    number_overlap = c_numbers & e_numbers
    number_mismatch = bool(c_numbers and e_numbers and not number_overlap)

    lexical_match = len(shared) >= 3 and overlap_ratio >= 0.20
    if name_overlap or lexical_match:
        heuristic_status = "likely_relevant"
    elif len(shared) >= 2:
        heuristic_status = "uncertain"
    else:
        heuristic_status = "likely_unrelated"

    return {
        "relevance_status": heuristic_status,
        "relevance_shared_token_count": len(shared),
        "relevance_shared_tokens": "; ".join(sorted(shared)),
        "relevance_token_overlap_ratio": round(overlap_ratio, 4),
        "relevance_shared_entity_count": len(name_overlap),
        "relevance_shared_entities": "; ".join(sorted(name_overlap)),
        "relevance_entity_jaccard": round(entity_jaccard, 4),
        "relevance_lexical_match": int(lexical_match),
        "relevance_number_mismatch_flag": int(number_mismatch),
    }


def arbitrate_relevance(heuristic: dict, model_score: float | None,
                        model_threshold: float = 0.50,
                        min_shared_entities: int = 1) -> dict:
    """Combine neural relevance with independent entity/lexical checks.

    A high neural score passes only when at least one meaningful entity match or
    sufficient lexical overlap exists. A low neural score that conflicts with
    strong heuristic evidence is marked uncertain rather than silently rejected.
    """
    if model_score is None or not math.isfinite(float(model_score)):
        return {
            **heuristic,
            "heuristic_relevance_status": heuristic["relevance_status"],
            "relevance_model_score": None,
            "relevance_arbitration_reason": "heuristic_only_no_model_score",
        }

    score = float(model_score)
    entity_match = int(heuristic.get("relevance_shared_entity_count", 0)) >= min_shared_entities
    lexical_match = bool(heuristic.get("relevance_lexical_match", 0))
    heuristic_status = heuristic["relevance_status"]

    if score >= model_threshold and (entity_match or lexical_match):
        status, reason = "likely_relevant", (
            "neural_plus_entity_match" if entity_match else "neural_plus_lexical_match"
        )
    elif score >= model_threshold:
        if heuristic_status == "uncertain":
            status, reason = "uncertain", "high_neural_score_weak_heuristic_overlap"
        else:
            status, reason = "likely_unrelated", "high_neural_score_without_entity_or_lexical_match"
    elif entity_match or lexical_match:
        status, reason = "uncertain", "low_neural_score_conflicts_with_heuristic_match"
    elif heuristic_status == "uncertain":
        status, reason = "uncertain", "low_neural_score_and_weak_lexical_overlap"
    else:
        status, reason = "likely_unrelated", "low_neural_and_heuristic_relevance"

    return {
        **heuristic,
        "heuristic_relevance_status": heuristic_status,
        "relevance_model_score": score,
        "relevance_status": status,
        "relevance_arbitration_reason": reason,
    }


def load_pairs(conn: sqlite3.Connection, limit: int | None = None,
               include_stale: bool = False) -> pd.DataFrame:
    """Load candidate/passage text and metadata; fail clearly on incompatible DBs."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required_tables = {"claim_candidates", "evidence_passages", "claim_evidence"}
    missing_tables = required_tables - tables
    if missing_tables:
        raise RuntimeError(
            "Missing retrieval tables: " + ", ".join(sorted(missing_tables)) +
            ". Run the evidence retrieval stage against this database first."
        )
    schemas = {}
    for table in required_tables:
        schemas[table] = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    required = {
        "claim_candidates": {"claim_id", "claim_text", "origin_article_id"},
        "evidence_passages": {"passage_id", "article_id", "passage_text"},
        "claim_evidence": {"claim_id", "passage_id", "retrieval_rank"},
    }
    for table, cols in required.items():
        missing = cols - schemas[table]
        if missing:
            raise RuntimeError(f"{table} is missing required columns: {', '.join(sorted(missing))}")
    missing_current = [
        f"{table}.is_current"
        for table in ("claim_candidates", "claim_evidence")
        if "is_current" not in schemas[table]
    ]
    if missing_current and not include_stale:
        raise RuntimeError(
            "Current-run tracking is not initialized (missing "
            + ", ".join(missing_current)
            + "). Run the updated evidence retrieval stage first, or use --include-stale "
            "only when historical scoring is intentional."
        )
    current_filter = "" if include_stale else "AND COALESCE(ce.is_current, 1) = 1"
    candidate_current_filter = (
        "" if include_stale
        else "AND COALESCE(cc.is_current, 1) = 1"
    )
    def optional_expr(table: str, sql_alias: str, column: str, output: str,
                      default_sql: str = "''") -> str:
        return (
            f"{sql_alias}.{column} AS {output}"
            if column in schemas.get(table, set())
            else f"{default_sql} AS {output}"
        )

    article_cols = set()
    if "articles" in tables:
        article_cols = {r[1] for r in conn.execute('PRAGMA table_info("articles")')}
    evidence_url_expr = "a.url AS evidence_url" if "url" in article_cols else "'' AS evidence_url"
    evidence_domain_expr = "a.domain AS evidence_domain" if "domain" in article_cols else "'' AS evidence_domain"
    evidence_source_expr = "a.source_name AS evidence_source_name" if "source_name" in article_cols else ("a.source AS evidence_source_name" if "source" in article_cols else "'' AS evidence_source_name")
    article_join = "LEFT JOIN articles a ON a.article_id = ep.article_id" if "article_id" in article_cols else ""
    claim_title_expr = optional_expr("claim_candidates", "cc", "origin_title", "claim_title")
    claim_url_expr = optional_expr("claim_candidates", "cc", "origin_url", "claim_url")
    claim_domain_expr = optional_expr("claim_candidates", "cc", "origin_domain", "claim_domain")
    claim_date_expr = optional_expr("claim_candidates", "cc", "origin_published_utc", "claim_published_utc")
    cluster_expr = optional_expr("claim_evidence", "ce", "same_event_cluster", "same_event_cluster", "0")
    similarity_expr = optional_expr("claim_evidence", "ce", "similarity_score", "similarity_score", "NULL")
    shared_entities_expr = optional_expr("claim_evidence", "ce", "shared_entities", "stored_shared_entities")
    assessment_expr = optional_expr("claim_evidence", "ce", "assessment", "human_assessment", "'unassessed'")
    query = f"""
        SELECT ce.claim_id, ce.passage_id, ce.retrieval_rank,
               cc.claim_text, cc.origin_article_id, {claim_title_expr},
               {claim_url_expr}, {claim_domain_expr}, {claim_date_expr},
               ep.article_id AS evidence_article_id, ep.passage_text AS evidence_text,
               {evidence_url_expr}, {evidence_domain_expr}, {evidence_source_expr},
               {cluster_expr}, {similarity_expr}, {shared_entities_expr}, {assessment_expr}
        FROM claim_evidence ce
        JOIN claim_candidates cc ON cc.claim_id = ce.claim_id
        JOIN evidence_passages ep ON ep.passage_id = ce.passage_id
        {article_join}
        WHERE 1=1 {current_filter} {candidate_current_filter}
        ORDER BY ce.claim_id, ce.retrieval_rank, ce.passage_id
    """
    frame = pd.read_sql_query(query, conn)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        frame = frame.head(limit)
    if frame.empty:
        LOG.warning("No claim-evidence pairs found to score.")
    return frame


def _normalize_label(label: str) -> str | None:
    s = re.sub(r"[^a-z]", "", str(label).lower())
    if any(x in s for x in ("entail", "support")):
        return "supports"
    if any(x in s for x in ("contradict", "refut")):
        return "contradicts"
    if any(x in s for x in ("neutral", "notenough", "unknown")):
        return "insufficient"
    return None


def resolve_label_map(config) -> dict[int, str]:
    """Resolve model labels by semantics, refusing ambiguous LABEL_0-only configs."""
    raw = getattr(config, "id2label", None) or {}
    resolved = {}
    for key, value in raw.items():
        stance = _normalize_label(value)
        if stance:
            resolved[int(key)] = stance
    expected = {"supports", "contradicts", "insufficient"}
    if set(resolved.values()) != expected or set(resolved.keys()) != {0, 1, 2}:
        raise RuntimeError(
            "Could not safely map this model's id2label to entailment/contradiction/neutral. "
            f"Model labels were: {raw}. Refusing to guess label order."
        )
    return resolved



def load_relevance_model(model_name: str, device: str | None = None):
    """Load the fine-tuned one-logit relevance cross-encoder."""
    if not model_name:
        return None, None, None, None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Relevance-model dependencies are missing. Install requirements-nli.txt."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if int(getattr(model.config, "num_labels", -1)) != 1:
        raise RuntimeError(
            f"Expected a one-logit relevance checkpoint, found num_labels={model.config.num_labels}."
        )
    if device:
        target = device
    else:
        target = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if target == "mps" and not torch.backends.mps.is_available():
        target = "cpu"
    if target == "cuda" and not torch.cuda.is_available():
        target = "cpu"
    model.to(target)
    model.eval()
    return tokenizer, model, target, torch


def score_relevance_pairs(frame: pd.DataFrame, tokenizer, model, device, torch,
                          batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> np.ndarray:
    """Return sigmoid relevance scores; these are not calibrated probabilities."""
    scores = np.full(len(frame), np.nan, dtype=np.float32)
    if frame.empty:
        return scores
    claims = frame["claim_text"].fillna("").astype(str).tolist()
    evidence = frame["evidence_text"].fillna("").astype(str).tolist()
    valid = [i for i, (c, e) in enumerate(zip(claims, evidence)) if c.strip() and e.strip()]
    for start in range(0, len(valid), batch_size):
        positions = valid[start:start + batch_size]
        encoded = tokenizer(
            [claims[i] for i in positions], [evidence[i] for i in positions],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError(f"Unexpected relevance logits shape: {tuple(logits.shape)}")
            batch_scores = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
        if not np.isfinite(batch_scores).all():
            raise RuntimeError("Relevance model produced non-finite scores.")
        for i, score in zip(positions, batch_scores):
            scores[i] = float(score)
    return scores


def load_model(model_name: str, device: str | None = None):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "NLI dependencies are missing. Install requirements-nli.txt in your project venv."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    label_map = resolve_label_map(model.config)
    if device:
        target = device
    else:
        target = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if target == "mps" and not torch.backends.mps.is_available():
        LOG.warning("MPS unavailable; falling back to CPU")
        target = "cpu"
    if target == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA unavailable; falling back to CPU")
        target = "cpu"
    model.to(target)
    model.eval()
    return tokenizer, model, label_map, target, torch


def score_pairs(frame: pd.DataFrame, tokenizer, model, label_map: dict[int, str],
                device: str, torch, model_name: str, batch_size: int = BATCH_SIZE,
                max_length: int = MAX_LENGTH, thresholds: NLIThresholds = NLIThresholds(),
                force_abstain_same_article: bool = True,
                relevance_scores: np.ndarray | None = None,
                relevance_threshold: float = 0.50,
                min_shared_entities: int = 1) -> pd.DataFrame:
    validate_thresholds(thresholds)
    if batch_size < 1 or max_length < 32:
        raise ValueError("batch_size must be >=1 and max_length must be >=32")
    required = {"claim_id", "passage_id", "claim_text", "evidence_text", "origin_article_id", "evidence_article_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Input frame missing: " + ", ".join(sorted(missing)))
    if frame.empty:
        return pd.DataFrame(columns=[
            "claim_id", "passage_id", "model_name", "model_version", "claim_text",
            "evidence_text", "claim_url", "claim_domain", "evidence_url",
            "evidence_domain", "retrieval_rank", "same_article", "scored_utc",
            "pair_id", "status", "predicted_stance", "raw_top_stance",
            "p_supports", "p_contradicts", "p_insufficient", "confidence", "margin",
            "relevance_status", "relevance_model_score", "heuristic_relevance_status",
            "relevance_arbitration_reason", "relevance_shared_token_count", "relevance_shared_tokens",
            "relevance_token_overlap_ratio", "relevance_shared_entity_count",
            "relevance_shared_entities", "relevance_entity_jaccard", "relevance_lexical_match",
            "relevance_number_mismatch_flag",
            "verdict_eligible", "interpretation"
        ])
    records = []
    valid_positions = []
    premises, hypotheses = [], []
    gate_by_idx = {}
    frame = frame.reset_index(drop=True)
    if not 0.0 <= relevance_threshold <= 1.0:
        raise ValueError("relevance_threshold must be between 0 and 1")
    if min_shared_entities < 1:
        raise ValueError("min_shared_entities must be >= 1")
    for idx, row in frame.iterrows():
        claim = clean_text(row.claim_text)
        evidence = clean_text(row.evidence_text)
        if not claim or not evidence:
            gate_by_idx[idx] = {
                "relevance_status": "empty_text",
                "relevance_model_score": None,
                "heuristic_relevance_status": "empty_text",
                "relevance_arbitration_reason": "empty_claim_or_evidence_text",
            }
            continue
        heuristic = relevance_gate(claim, evidence, row.get("stored_shared_entities", ""))
        model_score = None
        if relevance_scores is not None and idx < len(relevance_scores) and np.isfinite(relevance_scores[idx]):
            model_score = float(relevance_scores[idx])
        gate = arbitrate_relevance(
            heuristic, model_score, relevance_threshold, min_shared_entities
        )
        gate_by_idx[idx] = gate
        # Only concordant neural + heuristic evidence proceeds to stance inference.
        if gate["relevance_status"] != "likely_relevant":
            continue
        premises.append(evidence)
        hypotheses.append(claim)
        valid_positions.append(idx)
    logits_by_idx = {}
    for start in range(0, len(valid_positions), batch_size):
        end = start + batch_size
        encoded = tokenizer(
            premises[start:end], hypotheses[start:end], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != len(label_map):
                raise RuntimeError(
                    f"Unexpected model logits shape {tuple(logits.shape)}; "
                    f"expected [batch, {len(label_map)}]."
                )
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
        if not np.isfinite(probs).all():
            raise RuntimeError("Model produced non-finite probabilities.")
        for offset, vector in enumerate(probs):
            logits_by_idx[valid_positions[start + offset]] = vector
    for idx, row in frame.reset_index(drop=True).iterrows():
        claim = clean_text(row.claim_text)
        evidence = clean_text(row.evidence_text)
        same_article = bool(clean_text(row.origin_article_id) and
                            clean_text(row.origin_article_id) == clean_text(row.evidence_article_id))
        gate = gate_by_idx.get(idx, {"relevance_status": "empty_text"})
        base = {
            "claim_id": clean_text(row.claim_id), "passage_id": clean_text(row.passage_id),
            **gate,
            "model_name": model_name,
            "model_version": getattr(getattr(model, "config", None), "_commit_hash", None) or "revision-not-reported", 
            "claim_text": claim, "evidence_text": evidence,
            "claim_url": clean_text(row.get("claim_url", "")),
            "claim_domain": clean_text(row.get("claim_domain", "")),
            "evidence_url": clean_text(row.get("evidence_url", "")),
            "evidence_domain": clean_text(row.get("evidence_domain", "")),
            "retrieval_rank": (
                int(row.get("retrieval_rank"))
                if pd.notna(row.get("retrieval_rank")) and str(row.get("retrieval_rank")).strip() != ""
                else None
            ),
            "same_article": int(same_article),
            "scored_utc": datetime.now(timezone.utc).isoformat(),
            "pair_id": stable_pair_id(clean_text(row.claim_id), clean_text(row.passage_id), model_name),
        }
        if idx not in logits_by_idx:
            gate_status = gate.get("relevance_status")
            if gate_status == "empty_text":
                status = "skipped_empty_text"
            elif gate_status == "likely_unrelated":
                status = "skipped_likely_unrelated"
            else:
                status = "skipped_relevance_uncertain"
            base.update({"status": status, "predicted_stance": "abstain",
                         "raw_top_stance": None,
                         "p_supports": None, "p_contradicts": None, "p_insufficient": None,
                         "confidence": None, "margin": None,
                         "verdict_eligible": 0,
                         "interpretation": "No NLI verdict: relevance gate did not pass."})
            records.append(base)
            continue
        vector = logits_by_idx[idx]
        probs = {label_map[i]: float(vector[i]) for i in range(len(vector))}
        if set(probs) != {"supports", "contradicts", "insufficient"}:
            raise RuntimeError(f"Unexpected normalized label set: {sorted(probs)}")
        ordered = sorted(probs.items(), key=lambda x: (-x[1], x[0]))
        raw_stance, confidence = ordered[0]
        margin = confidence - ordered[1][1]
        if force_abstain_same_article and same_article:
            predicted, status = "abstain", "abstained_same_article"
        elif confidence < thresholds.min_confidence or margin < thresholds.min_margin:
            predicted, status = "abstain", "low_confidence_or_margin"
        elif raw_stance == "supports" and confidence < thresholds.min_entailment:
            predicted, status = "abstain", "below_support_threshold"
        elif raw_stance == "contradicts" and confidence < thresholds.min_contradiction:
            predicted, status = "abstain", "below_contradiction_threshold"
        else:
            predicted, status = raw_stance, "predicted"
        base.update({
            "status": "model_signal_only" if predicted != "abstain" else status,
            "predicted_stance": predicted,
            "raw_top_stance": raw_stance,
            "p_supports": probs["supports"], "p_contradicts": probs["contradicts"],
            "p_insufficient": probs["insufficient"], "confidence": confidence,
            "margin": margin,
            "verdict_eligible": 0,
            "interpretation": "Experimental NLI signal only; not a verified support/contradiction verdict.",
        })
        records.append(base)
    return pd.DataFrame(records).drop(columns=["_idx"], errors="ignore")


def persist_results(conn: sqlite3.Connection, results: pd.DataFrame) -> None:
    """Upsert model predictions separately; never alter human assessment columns."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_evidence_nli (
            pair_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            passage_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT,
            status TEXT NOT NULL,
            predicted_stance TEXT NOT NULL,
            raw_top_stance TEXT,
            p_supports REAL,
            p_contradicts REAL,
            p_insufficient REAL,
            confidence REAL,
            margin REAL,
            same_article INTEGER NOT NULL DEFAULT 0,
            scored_utc TEXT NOT NULL,
            relevance_status TEXT,
            relevance_model_score REAL,
            heuristic_relevance_status TEXT,
            relevance_arbitration_reason TEXT,
            relevance_shared_token_count INTEGER,
            relevance_shared_tokens TEXT,
            relevance_token_overlap_ratio REAL,
            relevance_shared_entity_count INTEGER,
            relevance_shared_entities TEXT,
            relevance_entity_jaccard REAL,
            relevance_lexical_match INTEGER,
            relevance_number_mismatch_flag INTEGER,
            verdict_eligible INTEGER NOT NULL DEFAULT 0,
            interpretation TEXT,
            UNIQUE(claim_id, passage_id, model_name)
        )
    """)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(claim_evidence_nli)")}
    migrations = {
        "relevance_status": "TEXT",
        "relevance_model_score": "REAL",
        "heuristic_relevance_status": "TEXT",
        "relevance_arbitration_reason": "TEXT",
        "relevance_shared_token_count": "INTEGER",
        "relevance_shared_tokens": "TEXT",
        "relevance_token_overlap_ratio": "REAL",
        "relevance_shared_entity_count": "INTEGER",
        "relevance_shared_entities": "TEXT",
        "relevance_entity_jaccard": "REAL",
        "relevance_lexical_match": "INTEGER",
        "relevance_number_mismatch_flag": "INTEGER",
        "verdict_eligible": "INTEGER NOT NULL DEFAULT 0",
        "interpretation": "TEXT",
    }
    for column, declaration in migrations.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE claim_evidence_nli ADD COLUMN {column} {declaration}")
    if results.empty:
        conn.commit()
        return
    sql = """
        INSERT INTO claim_evidence_nli (
            pair_id,claim_id,passage_id,model_name,model_version,status,predicted_stance,
            raw_top_stance,p_supports,p_contradicts,p_insufficient,confidence,margin,
            same_article,scored_utc,relevance_status,relevance_model_score,heuristic_relevance_status,
            relevance_arbitration_reason,relevance_shared_token_count,relevance_shared_tokens,
            relevance_token_overlap_ratio,relevance_shared_entity_count,relevance_shared_entities,
            relevance_entity_jaccard,relevance_lexical_match,relevance_number_mismatch_flag,
            verdict_eligible,interpretation
        ) VALUES (
            :pair_id,:claim_id,:passage_id,:model_name,:model_version,:status,:predicted_stance,
            :raw_top_stance,:p_supports,:p_contradicts,:p_insufficient,:confidence,:margin,
            :same_article,:scored_utc,:relevance_status,:relevance_model_score,:heuristic_relevance_status,
            :relevance_arbitration_reason,:relevance_shared_token_count,:relevance_shared_tokens,
            :relevance_token_overlap_ratio,:relevance_shared_entity_count,:relevance_shared_entities,
            :relevance_entity_jaccard,:relevance_lexical_match,:relevance_number_mismatch_flag,
            :verdict_eligible,:interpretation
        ) ON CONFLICT(claim_id,passage_id,model_name) DO UPDATE SET
            pair_id=excluded.pair_id,model_version=excluded.model_version,status=excluded.status,
            predicted_stance=excluded.predicted_stance,raw_top_stance=excluded.raw_top_stance,
            p_supports=excluded.p_supports,p_contradicts=excluded.p_contradicts,
            p_insufficient=excluded.p_insufficient,confidence=excluded.confidence,
            margin=excluded.margin,same_article=excluded.same_article,scored_utc=excluded.scored_utc,
            relevance_status=excluded.relevance_status,
            relevance_model_score=excluded.relevance_model_score,
            heuristic_relevance_status=excluded.heuristic_relevance_status,
            relevance_arbitration_reason=excluded.relevance_arbitration_reason,
            relevance_shared_token_count=excluded.relevance_shared_token_count,
            relevance_shared_tokens=excluded.relevance_shared_tokens,
            relevance_token_overlap_ratio=excluded.relevance_token_overlap_ratio,
            relevance_shared_entity_count=excluded.relevance_shared_entity_count,
            relevance_shared_entities=excluded.relevance_shared_entities,
            relevance_entity_jaccard=excluded.relevance_entity_jaccard,
            relevance_lexical_match=excluded.relevance_lexical_match,
            relevance_number_mismatch_flag=excluded.relevance_number_mismatch_flag,
            verdict_eligible=0,interpretation=excluded.interpretation
    """
    keep = ["pair_id","claim_id","passage_id","model_name","model_version","status",
            "predicted_stance","raw_top_stance","p_supports","p_contradicts","p_insufficient",
            "confidence","margin","same_article","scored_utc",
            "relevance_status","relevance_model_score","heuristic_relevance_status",
            "relevance_arbitration_reason","relevance_shared_token_count","relevance_shared_tokens",
            "relevance_token_overlap_ratio","relevance_shared_entity_count",
            "relevance_shared_entities","relevance_entity_jaccard","relevance_lexical_match",
            "relevance_number_mismatch_flag","verdict_eligible","interpretation"]
    rows = results[keep].replace({np.nan: None}).to_dict("records")
    conn.executemany(sql, rows)
    conn.commit()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tmp",
                                         dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            frame.to_csv(f, index=False)
        tmp.replace(path)
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


def run_diagnostic(model_name: str, device: str | None, batch_size: int,
                   max_length: int, thresholds: NLIThresholds) -> int:
    """Run fixed sanity pairs without requiring a project database."""
    tokenizer, model, label_map, actual_device, torch = load_model(model_name, device)
    examples = [
        ("support_1", "The government declared a state of emergency across the country.",
         "The government announced a nationwide state of emergency.", "supports"),
        ("support_2", "The minister confirmed that the bill was approved by parliament.",
         "Parliament approved the bill.", "supports"),
        ("contradiction_1", "The government confirmed that no state of emergency had been declared.",
         "The government announced a nationwide state of emergency.", "contradicts"),
        ("contradiction_2", "The bill was rejected by parliament.",
         "Parliament approved the bill.", "contradicts"),
        ("neutral_economy", "The country's economy grew by 3 percent last year.",
         "The government announced a nationwide state of emergency.", "insufficient"),
        ("neutral_fish", "Scientists discovered a new species of marine fish.",
         "The government announced a nationwide state of emergency.", "insufficient"),
        ("neutral_flood", "Heavy rainfall caused flooding in several coastal villages.",
         "The government announced a nationwide state of emergency.", "insufficient"),
        ("neutral_company", "The company reported record quarterly revenue.",
         "The government announced a nationwide state of emergency.", "insufficient"),
        ("neutral_rail", "A new railway line opened in the capital.",
         "The government announced a nationwide state of emergency.", "insufficient"),
        ("neutral_population", "The country's population increased during the year.",
         "The government announced a nationwide state of emergency.", "insufficient"),
    ]
    frame = pd.DataFrame([
        {
            "claim_id": f"diagnostic_{name}",
            "passage_id": f"passage_{name}",
            "origin_article_id": f"claim_article_{name}",
            "evidence_article_id": f"evidence_article_{name}",
            "claim_text": hypothesis,
            "evidence_text": premise,
            "retrieval_rank": 1,
        }
        for name, premise, hypothesis, _ in examples
    ])
    result = score_pairs(
        frame, tokenizer, model, label_map, actual_device, torch, model_name,
        batch_size=batch_size, max_length=max_length, thresholds=thresholds,
        force_abstain_same_article=False,
    )
    expected = {f"diagnostic_{name}": label for name, _, _, label in examples}
    result["expected_stance"] = result["claim_id"].map(expected)
    result["diagnostic_match"] = result["raw_top_stance"] == result["expected_stance"]
    print(f"Model: {model_name}")
    print(f"Device: {actual_device}")
    print(f"Label mapping: {label_map}")
    cols = [
        "claim_id", "expected_stance", "raw_top_stance", "predicted_stance",
        "p_supports", "p_contradicts", "p_insufficient", "confidence", "margin",
        "status", "diagnostic_match",
    ]
    print(result[cols].to_string(index=False))
    neutral = result[result["expected_stance"] == "insufficient"]
    false_contradictions = int((neutral["raw_top_stance"] == "contradicts").sum())
    print(
        f"\nSanity summary: {int(result['diagnostic_match'].sum())}/{len(result)} "
        f"top-label matches; {false_contradictions}/{len(neutral)} neutral controls "
        "were classified as contradiction."
    )
    print("This is a small diagnostic set, not a model accuracy benchmark.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Three-way stance checkpoint; defaults to local fine-tuned DeBERTa when present.")
    parser.add_argument("--relevance-model", default=DEFAULT_RELEVANCE_MODEL,
                        help="One-logit relevance checkpoint; defaults to local fine-tuned MiniLM when present.")
    parser.add_argument("--relevance-threshold", type=float, default=0.50,
                        help="Sigmoid relevance-score threshold (not a calibrated probability).")
    parser.add_argument("--min-shared-entities", type=int, default=1,
                        help="Minimum shared entity candidates required to pass alongside a high neural score.")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run fixed support/contradiction/neutral sanity pairs; no DB required")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--limit", type=int, help="Score only first N pairs (smoke test)")
    parser.add_argument("--include-stale", action="store_true", help="Also score retrieval links marked is_current=0")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--min-entailment", type=float, default=0.70)
    parser.add_argument("--min-contradiction", type=float, default=0.70)
    parser.add_argument("--allow-same-article", action="store_true", help="Do not force abstention for same-article pairs")
    args = parser.parse_args(argv)
    thresholds = NLIThresholds(args.min_confidence, args.min_margin,
                               args.min_entailment, args.min_contradiction)
    validate_thresholds(thresholds)
    if args.device == "auto":
        device = None
    else:
        device = args.device
    if args.diagnostic:
        return run_diagnostic(args.model, device, args.batch_size, args.max_length, thresholds)
    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")
    with sqlite3.connect(args.db) as conn:
        pairs = load_pairs(conn, args.limit, args.include_stale)
        if pairs.empty:
            LOG.warning("No current claim-evidence pairs; writing an empty current-run NLI report.")
            results = pd.DataFrame(columns=[
                "claim_id", "passage_id", "model_name", "model_version", "claim_text",
                "evidence_text", "claim_url", "claim_domain", "evidence_url",
                "evidence_domain", "retrieval_rank", "same_article", "scored_utc",
                "pair_id", "status", "predicted_stance", "raw_top_stance",
                "p_supports", "p_contradicts", "p_insufficient", "confidence", "margin",
                "relevance_status", "relevance_model_score", "heuristic_relevance_status",
                "relevance_arbitration_reason", "relevance_shared_token_count", "relevance_shared_tokens",
                "relevance_token_overlap_ratio", "relevance_shared_entity_count",
                "relevance_shared_entities", "relevance_entity_jaccard", "relevance_lexical_match",
                "relevance_number_mismatch_flag",
                "verdict_eligible", "interpretation",
            ])
        else:
            tokenizer, model, label_map, actual_device, torch = load_model(args.model, device)
            relevance_scores = None
            if args.relevance_model:
                rel_tokenizer, rel_model, rel_device, rel_torch = load_relevance_model(args.relevance_model, device)
                LOG.info("Scoring relevance for %d pairs with %s on %s",
                         len(pairs), args.relevance_model, rel_device)
                relevance_scores = score_relevance_pairs(
                    pairs, rel_tokenizer, rel_model, rel_device, rel_torch,
                    batch_size=args.batch_size, max_length=args.max_length,
                )
            else:
                LOG.warning("No relevance checkpoint supplied; using lexical/entity relevance gate.")
            LOG.info("Scoring stance for %d pairs with %s on %s", len(pairs), args.model, actual_device)
            results = score_pairs(
                pairs, tokenizer, model, label_map, actual_device, torch, args.model,
                args.batch_size, args.max_length, thresholds,
                force_abstain_same_article=not args.allow_same_article,
                relevance_scores=relevance_scores,
                relevance_threshold=args.relevance_threshold,
                min_shared_entities=args.min_shared_entities,
            )
        persist_results(conn, results)
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "claim_evidence_nli_review.csv"
    atomic_csv(results, out)
    counts = results["predicted_stance"].value_counts(dropna=False).to_dict()
    LOG.info("Saved %s", out)
    LOG.info("Prediction counts (includes abstentions): %s", counts)
    LOG.info("NLI outputs are experimental triage signals, never factual verdicts.")
    LOG.info("Pairs failing the relevance gate are abstained before NLI inference.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("NLI stage failed: %s", exc)
        raise
