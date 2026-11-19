#!/usr/bin/env python3
"""
Claim candidate extraction and evidence retrieval for the OSINT event pipeline.

This is a retrieval/review stage, NOT an automatic fact checker.
- Candidate sentences are heuristic suggestions, not validated atomic claims.
- Hybrid retrieval is the default: TF-IDF + sentence-transformer rankings
  are fused with Reciprocal Rank Fusion (RRF).
- TF-IDF-only retrieval remains available with --retriever tfidf.
- Evidence assessments remain human-reviewed / unassessed.
- Existing human assessments and review notes are preserved on reruns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "new_articles.db"
DEFAULT_REPORTS = ROOT / "reports"

MIN_CHARS = 55
MAX_CHARS = 550
DEFAULT_MAX_CLAIMS_PER_ARTICLE = 20
DEFAULT_MAX_PASSAGES_PER_ARTICLE = 250
DEFAULT_REVIEW_SAMPLE_CLAIMS = 100
DEFAULT_MIN_LEXICAL_SCORE = 0.02
DEFAULT_MIN_SEMANTIC_SCORE = 0.35
RRF_K = 60

# Specific factual/action cues are weighted more heavily than generic auxiliaries.
STRONG_CUES = re.compile(
    r"\b(killed|injured|wounded|died|arrested|detained|charged|"
    r"sentenced|convicted|acquitted|filed|approved|rejected|"
    r"elected|resigned|voted|signed|launched|struck|attacked|"
    r"exploded|collapsed|evacuated|captured|released|withdrew|"
    r"reached|increased|decreased|displaced|sanctioned|"
    r"announced|confirmed|denied|warned|accused|claimed|stated)\b",
    re.IGNORECASE,
)
ATTRIBUTION_CUES = re.compile(
    r"\b(said|says|according to|reported|announced|confirmed|"
    r"denied|claimed|stated|warned|accused)\b", re.IGNORECASE
)
GENERIC_ONLY = re.compile(
    r"^(this|that|it|they|he|she|we|you)\s+(is|was|are|were|has|have|had)\b",
    re.IGNORECASE,
)
BOILERPLATE = re.compile(
    r"\b(subscribe|sign up|newsletter|read more|click here|"
    r"follow us|share this|advertisement|all rights reserved|"
    r"privacy policy|terms of use)\b", re.IGNORECASE,
)
NUMBER_OR_DATE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b", re.IGNORECASE
)


def clean(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value) -> str:
    value = clean(value).lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def make_id(*parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def split_compound_segments(sentence):
    """Split obvious feed/page concatenations without guessing sentence meaning.

    A spaced vertical bar is used by several feeds/pages to join independent
    story snippets. Each side becomes its own passage and claim candidate.
    """
    value = clean(sentence)
    if not value:
        return []
    parts = [clean(part) for part in re.split(r"\s+[|¦]\s+", value)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [value]
    return parts


def parse_sentences(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else None
    except (ValueError, TypeError):
        data = None
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, str):
            value = clean(item)
        elif isinstance(item, dict):
            value = clean(item.get("text") or item.get("sentence") or item.get("content"))
        else:
            value = ""
        if value:
            result.extend(split_compound_segments(value))
    return result


def fallback_split(body):
    # Conservative fallback; article_nlp.sentences_json is preferred.
    return [
        clean(x) for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", clean(body))
        if clean(x)
    ]


def candidate_score(sentence: str) -> int:
    """Heuristic priority score. It does not determine factuality."""
    s = clean(sentence)
    if not MIN_CHARS <= len(s) <= MAX_CHARS:
        return -1
    if s.endswith("?") or BOILERPLATE.search(s) or GENERIC_ONLY.search(s):
        return -1
    if not re.search(r"\b[A-Z][a-z]{2,}\b", s):
        return -1

    score = 0
    if STRONG_CUES.search(s):
        score += 3
    if ATTRIBUTION_CUES.search(s):
        score += 1
    if NUMBER_OR_DATE.search(s):
        score += 1
    if re.search(r"\b(in|at|near|from|across|within)\s+[A-Z]", s):
        score += 1
    if len(s.split()) >= 10:
        score += 1

    # Avoid sentences that only state a broad opinion or prediction.
    if re.search(r"\b(could|might|may|perhaps|possibly|probably)\b", s, re.I):
        score -= 1
    return score if score >= 2 else -1



def load_articles(conn):
    """Load article text and optional article-level entities without assuming schema extras."""
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "articles" not in tables or "article_nlp" not in tables:
        raise RuntimeError(
            "Expected 'articles' and 'article_nlp'. Run article enrichment first."
        )

    ac = {r[1] for r in conn.execute("PRAGMA table_info(articles)")}
    nc = {r[1] for r in conn.execute("PRAGMA table_info(article_nlp)")}
    required_a = {
        "article_id", "title", "url", "domain", "source_name",
        "published_utc", "cluster_id"
    }
    required_n = {"article_id", "body_text"}
    missing = (required_a - ac) | (required_n - nc)
    if missing:
        raise RuntimeError("Missing DB columns: " + ", ".join(sorted(missing)))

    sentences_col = (
        "n.sentences_json AS sentences_json"
        if "sentences_json" in nc else "NULL AS sentences_json"
    )
    entities_col = (
        "n.entities_json AS entities_json"
        if "entities_json" in nc else "NULL AS entities_json"
    )
    query = f"""
        SELECT a.article_id, a.title, a.url, a.domain, a.source_name,
               a.published_utc, a.cluster_id, n.body_text,
               {sentences_col}, {entities_col}
        FROM articles a
        JOIN article_nlp n ON n.article_id = a.article_id
        WHERE COALESCE(n.body_text, '') <> ''
        ORDER BY a.published_utc, a.article_id
    """
    return pd.read_sql_query(query, conn)


def normalize_domain(domain_value="", url_value="", source_name=""):
    """
    Normalize a publisher domain for diversification.
    Prefer a valid URL hostname; fall back to the supplied domain, then source name.
    Unknown publishers receive an article-specific key rather than sharing one bucket.
    """
    candidates = [clean(url_value), clean(domain_value)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = urlparse(candidate if "://" in candidate else "//" + candidate)
            host = (parsed.hostname or "").lower().strip(".")
            if host:
                if host.startswith("www."):
                    host = host[4:]
                return host
        except (ValueError, TypeError):
            continue

    source = re.sub(r"\s+", " ", clean(source_name)).strip().casefold()
    return f"source:{source}" if source else ""


def parse_entity_names(raw):
    """Best-effort parser for the project's optional entities_json field."""
    if raw is None:
        return set()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return set()
    if isinstance(raw, dict):
        # Common spaCy / custom export shapes.
        if isinstance(raw.get("entities"), list):
            raw = raw["entities"]
        else:
            raw = [raw]

    if not isinstance(raw, list):
        return set()

    ignored = {
        "government", "minister", "president", "official", "company",
        "people", "person", "country", "state", "city", "today"
    }
    found = set()
    for item in raw:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = (
                item.get("text") or item.get("entity")
                or item.get("name") or item.get("value") or ""
            )
        else:
            continue
        value = normalize(value)
        if len(value) >= 3 and value not in ignored:
            found.add(value)
    return found


def entities_mentioned_in_text(entity_names, text):
    """
    Keep only extracted article entities explicitly mentioned in this sentence.
    This prevents an entity elsewhere in an article from being credited to
    every sentence in that article.
    """
    normalized_text = normalize(text)
    if not normalized_text:
        return set()
    found = set()
    for entity in entity_names or []:
        normalized_entity = normalize(entity)
        if not normalized_entity:
            continue
        pattern = r"(?<!\w)" + re.escape(normalized_entity) + r"(?!\w)"
        if re.search(pattern, normalized_text):
            found.add(normalized_entity)
    return found


def create_records(df, max_claims_per_article, max_passages_per_article):
    claims, passages = [], []
    for _, row in df.iterrows():
        aid = clean(row.article_id)
        if not aid:
            continue
        title = clean(row.title)
        url = clean(row.url)
        source_name = clean(row.source_name)
        domain_key = normalize_domain(row.get("domain", ""), url, source_name)
        article_entities = parse_entity_names(row.get("entities_json"))

        sentences = parse_sentences(row.get("sentences_json"))
        if not sentences:
            sentences = fallback_split(row.get("body_text", ""))

        # Exact normalized deduplication within an article; preserve original order.
        unique_sentences = []
        seen = set()
        for sentence in sentences[:max_passages_per_article]:
            key = normalize(sentence)
            if key and key not in seen:
                unique_sentences.append(sentence)
                seen.add(key)

        ranked_candidates = []
        for idx, sentence in enumerate(unique_sentences):
            pid = make_id(aid, str(idx), sentence)
            passage = {
                "passage_id": pid,
                "article_id": aid,
                "passage_index": idx,
                "passage_text": sentence,
                "title": title,
                "url": url,
                "domain": clean(row.get("domain", "")) or domain_key,
                "domain_key": domain_key or f"unknown:{aid}",
                "source_name": source_name,
                "published_utc": clean(row.published_utc),
                "cluster_id": clean(row.cluster_id),
                "entity_names": sorted(entities_mentioned_in_text(article_entities, sentence)),
            }
            passages.append(passage)
            score = candidate_score(sentence)
            if score >= 0:
                ranked_candidates.append((score, idx, sentence, pid))

        ranked_candidates.sort(key=lambda x: (-x[0], x[1]))
        for score, idx, sentence, pid in ranked_candidates[:max_claims_per_article]:
            claims.append({
                "claim_id": make_id(aid, normalize(sentence)),
                "origin_article_id": aid,
                "origin_passage_id": pid,
                "claim_text": sentence,
                "extraction_method": "heuristic_priority_v2",
                "extraction_status": "candidate_unreviewed",
                "candidate_priority": score,
                "origin_title": title,
                "origin_url": url,
                "origin_domain": clean(row.get("domain", "")) or domain_key,
                "origin_published_utc": clean(row.published_utc),
                "cluster_id": clean(row.cluster_id),
                "entity_names": sorted(entities_mentioned_in_text(article_entities, sentence)),
            })
    return claims, passages


def semantic_embeddings(claim_texts, passage_texts, model_name, batch_size=64):
    """Encode texts in batches; normalized vectors make dot products cosine scores."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid retrieval requires sentence-transformers. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    claim_vectors = model.encode(
        claim_texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    )
    passage_vectors = model.encode(
        passage_texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    )
    return np.asarray(claim_vectors, dtype=np.float32), np.asarray(
        passage_vectors, dtype=np.float32
    )


def _top_indices(scores, limit):
    """Return deterministic descending top indices, including ties at the cutoff."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    # Treat non-finite scores as worst possible values; this avoids unstable ordering.
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    limit = min(int(limit), scores.size)

    if limit == scores.size:
        selected = np.arange(scores.size, dtype=np.int64)
    else:
        cutoff = np.partition(scores, scores.size - limit)[scores.size - limit]
        better = np.flatnonzero(scores > cutoff)
        tied = np.flatnonzero(scores == cutoff)
        needed = limit - better.size
        # Stable original-index choice for ties exactly at the candidate-pool boundary.
        selected = np.concatenate((better, tied[:needed]))

    return selected[np.lexsort((selected, -scores[selected]))]




def parse_datetime_utc(value):
    """Parse an ISO-like timestamp as UTC; return None when missing or invalid."""
    value = clean(value)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def semantic_entity_weight(
    shared_entity_count,
    claim_has_entities,
    passage_has_entities,
    one_entity_weight=0.10,
    missing_entity_weight=0.50,
):
    """
    Gate the semantic RRF contribution using exact named-entity intersections.
    Lexical retrieval remains independent and can recover candidates without
    entity overlap. Missing NER data receives a conservative fallback weight.
    """
    if not claim_has_entities or not passage_has_entities:
        return float(missing_entity_weight)
    count = max(0, int(shared_entity_count))
    if count == 0:
        return 0.0
    if count == 1:
        return float(one_entity_weight)
    return 1.0


def temporal_decay_factor(
    claim_date, evidence_date, half_life_days=30.0, grace_days=3.0
):
    """
    Softly decay ranking by publication-date gap after a short grace period.
    At grace_days + half_life_days, the factor is exactly 0.5.
    Missing/invalid dates disable the penalty (factor 1.0).
    """
    if half_life_days <= 0:
        raise ValueError("temporal_half_life_days must be > 0")
    claim_dt = parse_datetime_utc(claim_date)
    evidence_dt = parse_datetime_utc(evidence_date)
    if claim_dt is None or evidence_dt is None:
        return None, 1.0
    gap_days = abs((evidence_dt - claim_dt).total_seconds()) / 86400.0
    effective_gap = max(0.0, gap_days - max(0.0, float(grace_days)))
    factor = 1.0 / (
        1.0 + np.log1p(effective_gap / float(half_life_days)) / np.log(2.0)
    )
    return gap_days, float(factor)


def retrieve(
    claims, passages, top_k, min_score=None, retriever="hybrid",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_per_domain=2, candidate_pool=100, batch_size=64,
    same_cluster_bonus=0.0005, min_lexical_score=DEFAULT_MIN_LEXICAL_SCORE,
    min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE,
    max_per_article=1, entity_one_weight=0.10,
    entity_missing_weight=0.50, temporal_half_life_days=30.0,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_batch_size=32
):
    """
    Retrieve passages with TF-IDF or hybrid RRF.

    Diversity is enforced at two levels: max_per_article and max_per_domain.
    Event-cluster membership is a small ranking signal only; cross-cluster
    candidates remain eligible so contradictory evidence is not suppressed.
    Article-level shared entities are an auditable tie-break signal, not proof
    of relevance and not a hard filter.
    """
    if not claims or not passages:
        return []
    if retriever not in {"tfidf", "hybrid"}:
        raise ValueError("retriever must be 'tfidf' or 'hybrid'")
    # Backward compatibility for callers using the former shared threshold.
    if min_score is not None:
        min_lexical_score = float(min_score)
        min_semantic_score = float(min_score)
    if top_k < 1 or candidate_pool < top_k or batch_size < 1:
        raise ValueError("Invalid top_k, candidate_pool, or batch_size.")
    if max_per_article < 1 or max_per_domain < 1:
        raise ValueError("Diversity limits must be >= 1.")
    if not 0 <= min_lexical_score <= 1 or not -1 <= min_semantic_score <= 1:
        raise ValueError("Invalid lexical or semantic threshold.")
    if not 0.0 <= entity_one_weight <= 1.0:
        raise ValueError("entity_one_weight must be between 0 and 1.")
    if not 0.0 <= entity_missing_weight <= 1.0:
        raise ValueError("entity_missing_weight must be between 0 and 1.")
    if temporal_half_life_days <= 0:
        raise ValueError("temporal_half_life_days must be > 0.")

    passage_texts = [clean(p.get("passage_text")) for p in passages]
    claim_texts = [clean(c.get("claim_text")) for c in claims]
    if not any(passage_texts):
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        sublinear_tf=True, min_df=1, max_df=0.98
    )
    try:
        p_matrix = vectorizer.fit_transform(passage_texts)
        c_matrix = vectorizer.transform(claim_texts)
    except ValueError as exc:
        # All passages may contain only stop words / be empty.
        if "empty vocabulary" in str(exc).lower():
            return []
        raise

    claim_vectors = passage_vectors = None
    if retriever == "hybrid":
        claim_vectors, passage_vectors = semantic_embeddings(
            claim_texts, passage_texts, model_name, batch_size=batch_size
        )
        if claim_vectors.shape[0] != len(claims) or passage_vectors.shape[0] != len(passages):
            raise RuntimeError("Embedding count does not match claims/passages.")
        if claim_vectors.shape[1] != passage_vectors.shape[1]:
            raise RuntimeError("Claim and passage embedding dimensions do not match.")

    # Entity sets are grounded to each claim/passage sentence in create_records.
    passage_entities = [
        set(p.get("entity_names") or []) for p in passages
    ]

    links = []
    candidate_groups = []
    pool_size = max(top_k, candidate_pool)

    # Cross-encoder reranking: unlike bi-encoder retrieval, this model reads
    # each claim and candidate passage together. It is a ranking signal only,
    # not a factual-verification decision.
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    reranker = CrossEncoder(reranker_model)

    for ci, claim in enumerate(claims):
        lex_scores = cosine_similarity(c_matrix[ci], p_matrix).ravel()
        sem_scores = None
        if retriever == "hybrid":
            sem_scores = passage_vectors @ claim_vectors[ci]
            sem_scores = np.nan_to_num(sem_scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

        origin_id = clean(claim.get("origin_article_id"))
        eligible = np.fromiter(
            (i for i, p in enumerate(passages)
             if clean(p.get("article_id")) != origin_id),
            dtype=np.int64
        )
        if eligible.size == 0:
            continue

        # Generate lexical candidates with a low threshold to retain exact terms.
        lex_order_local = _top_indices(lex_scores[eligible], pool_size)
        lex_order = [
            int(eligible[j]) for j in lex_order_local
            if float(lex_scores[eligible[j]]) >= min_lexical_score
        ]
        lex_rank = {pi: rank for rank, pi in enumerate(lex_order, 1)}

        sem_rank = {}
        if retriever == "hybrid":
            sem_order_local = _top_indices(sem_scores[eligible], pool_size)
            sem_order = [
                int(eligible[j]) for j in sem_order_local
                if float(sem_scores[eligible[j]]) >= min_semantic_score
            ]
            sem_rank = {pi: rank for rank, pi in enumerate(sem_order, 1)}

        candidate_ids = sorted(set(lex_rank) | set(sem_rank))
        candidates = []
        claim_entities = set(claim.get("entity_names") or [])
        for pi in candidate_ids:
            passage = passages[pi]
            lex = float(lex_scores[pi])
            sem = float(sem_scores[pi]) if sem_scores is not None else None
            shared_entities = sorted(
                claim_entities.intersection(passage_entities[pi])
            )
            if retriever == "hybrid":
                lexical_component = (
                    1.0 / (RRF_K + lex_rank[pi]) if pi in lex_rank else 0.0
                )
                semantic_component = (
                    1.0 / (RRF_K + sem_rank[pi]) if pi in sem_rank else 0.0
                )
                entity_weight = semantic_entity_weight(
                    len(shared_entities),
                    bool(claim_entities),
                    bool(passage_entities[pi]),
                    one_entity_weight=entity_one_weight,
                    missing_entity_weight=entity_missing_weight,
                )
                fused = lexical_component + semantic_component * entity_weight
            else:
                entity_weight = 0.0
                fused = lex

            same_cluster = bool(
                clean(claim.get("cluster_id"))
                and clean(passage.get("cluster_id"))
                and clean(claim.get("cluster_id")) == clean(passage.get("cluster_id"))
            )
            pre_temporal_score = fused + (
                same_cluster_bonus if same_cluster else 0.0
            )
            temporal_gap_days, temporal_factor = temporal_decay_factor(
                claim.get("origin_published_utc"),
                passage.get("published_utc"),
                half_life_days=temporal_half_life_days,
            )
            rank_score = pre_temporal_score * temporal_factor
            candidates.append({
                "score": max(lex, sem) if sem is not None else lex,
                "rank_score": rank_score,
                "lexical": lex,
                "semantic": sem,
                "lexical_rank": lex_rank.get(pi),
                "semantic_rank": sem_rank.get(pi),
                "retrieval_methods": int(pi in lex_rank) + int(pi in sem_rank),
                "shared_entities": shared_entities,
                "shared_entity_count": len(shared_entities),
                "semantic_entity_weight": entity_weight,
                "temporal_gap_days": temporal_gap_days,
                "temporal_decay_factor": temporal_factor,
                "same_cluster": same_cluster,
                "passage": passage,
            })

        # Defer reranking until candidate pools for all claims have been built.
        # This avoids thousands of separate model.predict() calls.
        candidate_groups.append((claim, candidates))

    # Score all claim-passage pairs in batches across the complete run.
    flat_pairs = []
    group_ranges = []
    for claim, candidates in candidate_groups:
        group_start = len(flat_pairs)
        claim_text = clean(claim.get("claim_text"))
        flat_pairs.extend([
            [claim_text, clean(item["passage"].get("passage_text"))]
            for item in candidates
        ])
        group_ranges.append((claim, candidates, group_start, len(flat_pairs)))

    if flat_pairs:
        all_rerank_scores = reranker.predict(
            flat_pairs,
            batch_size=rerank_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        all_rerank_scores = np.asarray(all_rerank_scores).reshape(-1)
        if len(all_rerank_scores) != len(flat_pairs):
            raise RuntimeError("Cross-encoder returned an unexpected number of scores.")

        for claim, candidates, group_start, group_end in group_ranges:
            group_scores = all_rerank_scores[group_start:group_end]
            for item, score in zip(candidates, group_scores):
                item["reranker_score"] = float(score)

            # Cross-encoder score is primary; RRF and source/entity signals break ties.
            def tie_break_key(item):
                ranks = [
                    rank for rank in (item["lexical_rank"], item["semantic_rank"])
                    if rank is not None
                ]
                best_rank = min(ranks) if ranks else float("inf")
                worst_rank = max(ranks) if ranks else float("inf")
                semantic_score = (
                    item["semantic"] if item["semantic"] is not None else -1.0
                )
                return (
                    -item["reranker_score"],
                    -item["rank_score"],
                    -item["retrieval_methods"],
                    -item["shared_entity_count"],
                    best_rank,
                    worst_rank,
                    -semantic_score,
                    -item["lexical"],
                    clean(item["passage"].get("passage_id")),
                )

            candidates.sort(key=tie_break_key)

            # Strict article + publisher diversification. Never backfill by violating caps.
            selected, article_counts, domain_counts = [], defaultdict(int), defaultdict(int)
            for item in candidates:
                p = item["passage"]
                article_id = clean(p.get("article_id"))
                domain_key = clean(p.get("domain_key")) or normalize_domain(
                    p.get("domain", ""), p.get("url", ""), p.get("source_name", "")
                ) or f"unknown:{article_id or p.get('passage_id', '')}"

                if article_counts[article_id] >= max_per_article:
                    continue
                if domain_counts[domain_key] >= max_per_domain:
                    continue

                selected.append(item)
                article_counts[article_id] += 1
                domain_counts[domain_key] += 1
                if len(selected) >= top_k:
                    break

            for rank, item in enumerate(selected, 1):
                p = item["passage"]
                links.append({
                    "claim_id": claim["claim_id"],
                    "claim_text": claim.get("claim_text", ""),
                    "claim_title": claim.get("origin_title", ""),
                    "claim_url": claim.get("origin_url", ""),
                    "claim_domain": claim.get("origin_domain", ""),
                    "passage_id": p["passage_id"],
                    "retrieval_rank": rank,
                    "similarity_score": item["score"],
                    "lexical_score": item["lexical"],
                    "semantic_score": item["semantic"],
                    "rank_score": item["rank_score"],
                    "reranker_score": item["reranker_score"],
                    "lexical_rank": item["lexical_rank"],
                    "semantic_rank": item["semantic_rank"],
                    "retrieval_methods": item["retrieval_methods"],
                    "shared_entity_count": item["shared_entity_count"],
                    "shared_entities": "; ".join(item["shared_entities"]),
                    "semantic_entity_weight": item["semantic_entity_weight"],
                    "temporal_gap_days": item["temporal_gap_days"],
                    "temporal_decay_factor": item["temporal_decay_factor"],
                    "same_event_cluster": int(item["same_cluster"]),
                    "assessment": "unassessed",
                    "review_notes": "",
                    "evidence_article_id": p["article_id"],
                    "evidence_title": p["title"],
                    "evidence_url": p["url"],
                    "evidence_domain": p["domain"],
                    "evidence_domain_key": p.get("domain_key", ""),
                    "evidence_source_name": p["source_name"],
                    "evidence_published_utc": p["published_utc"],
                    "evidence_text": p["passage_text"],
                })
    return links


def ensure_column(conn, table, column, declaration):
    columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")



def _atomic_write_csv(frame, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp",
            prefix=destination.stem + "_", dir=destination.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        temp_path.replace(destination)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def persist(conn, claims, passages, links, reports, review_sample_claims=100, sample_seed=42):
    """
    Persist current retrieval results while preserving reviewer-owned fields.
    Old links are retained in SQLite and marked is_current=0 rather than deleted.
    CSV reports contain the current run only.
    """
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS claim_candidates (
        claim_id TEXT PRIMARY KEY, origin_article_id TEXT NOT NULL,
        origin_passage_id TEXT NOT NULL, claim_text TEXT NOT NULL,
        extraction_method TEXT NOT NULL, extraction_status TEXT NOT NULL,
        origin_title TEXT, origin_url TEXT, origin_domain TEXT,
        origin_published_utc TEXT, cluster_id TEXT
    );
    CREATE TABLE IF NOT EXISTS evidence_passages (
        passage_id TEXT PRIMARY KEY, article_id TEXT NOT NULL,
        passage_index INTEGER NOT NULL, passage_text TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS claim_evidence (
        claim_id TEXT NOT NULL, passage_id TEXT NOT NULL,
        retrieval_rank INTEGER NOT NULL, similarity_score REAL NOT NULL,
        same_event_cluster INTEGER NOT NULL,
        assessment TEXT NOT NULL DEFAULT 'unassessed',
        review_notes TEXT DEFAULT '',
        PRIMARY KEY (claim_id, passage_id)
    );
    """)

    for col, decl in [
        ("candidate_priority", "INTEGER"),
        ("is_current", "INTEGER NOT NULL DEFAULT 1"),
    ]:
        ensure_column(conn, "claim_candidates", col, decl)

    evidence_columns = [
        ("lexical_score", "REAL"),
        ("semantic_score", "REAL"),
        ("rank_score", "REAL"),
        ("reranker_score", "REAL"),
        ("lexical_rank", "INTEGER"),
        ("semantic_rank", "INTEGER"),
        ("retrieval_methods", "INTEGER"),
        ("shared_entity_count", "INTEGER"),
        ("shared_entities", "TEXT"),
        ("semantic_entity_weight", "REAL"),
        ("temporal_gap_days", "REAL"),
        ("temporal_decay_factor", "REAL"),
        ("is_current", "INTEGER NOT NULL DEFAULT 1"),
    ]
    for col, decl in evidence_columns:
        ensure_column(conn, "claim_evidence", col, decl)

    # This retrieval run covers the full article corpus. Mark every prior
    # candidate/link stale first, including IDs created by older extraction
    # algorithms. Current candidates/links are reactivated below. Reviewer-owned
    # assessment and notes columns are deliberately not modified on conflict.
    conn.execute("UPDATE claim_candidates SET is_current=0")
    conn.execute("UPDATE claim_evidence SET is_current=0")

    conn.executemany("""
        INSERT INTO claim_candidates (
            claim_id,origin_article_id,origin_passage_id,claim_text,
            extraction_method,extraction_status,origin_title,origin_url,
            origin_domain,origin_published_utc,cluster_id,candidate_priority,is_current
        ) VALUES (
            :claim_id,:origin_article_id,:origin_passage_id,:claim_text,
            :extraction_method,:extraction_status,:origin_title,:origin_url,
            :origin_domain,:origin_published_utc,:cluster_id,:candidate_priority,1
        )
        ON CONFLICT(claim_id) DO UPDATE SET
            origin_article_id=excluded.origin_article_id,
            origin_passage_id=excluded.origin_passage_id,
            claim_text=excluded.claim_text,
            extraction_method=excluded.extraction_method,
            extraction_status=excluded.extraction_status,
            origin_title=excluded.origin_title,
            origin_url=excluded.origin_url,
            origin_domain=excluded.origin_domain,
            origin_published_utc=excluded.origin_published_utc,
            cluster_id=excluded.cluster_id,
            candidate_priority=excluded.candidate_priority,
            is_current=1
    """, claims)

    conn.executemany("""
        INSERT INTO evidence_passages (passage_id,article_id,passage_index,passage_text)
        VALUES (:passage_id,:article_id,:passage_index,:passage_text)
        ON CONFLICT(passage_id) DO UPDATE SET
            article_id=excluded.article_id,
            passage_index=excluded.passage_index,
            passage_text=excluded.passage_text
    """, passages)

    if links:
        conn.executemany("""
            INSERT INTO claim_evidence (
                claim_id,passage_id,retrieval_rank,similarity_score,
                same_event_cluster,lexical_score,semantic_score,rank_score,reranker_score,
                lexical_rank,semantic_rank,retrieval_methods,
                shared_entity_count,shared_entities,semantic_entity_weight,
                temporal_gap_days,temporal_decay_factor,is_current,
                assessment,review_notes
            ) VALUES (
                :claim_id,:passage_id,:retrieval_rank,:similarity_score,
                :same_event_cluster,:lexical_score,:semantic_score,:rank_score,:reranker_score,
                :lexical_rank,:semantic_rank,:retrieval_methods,
                :shared_entity_count,:shared_entities,:semantic_entity_weight,
                :temporal_gap_days,:temporal_decay_factor,1,
                'unassessed',''
            )
            ON CONFLICT(claim_id,passage_id) DO UPDATE SET
                retrieval_rank=excluded.retrieval_rank,
                similarity_score=excluded.similarity_score,
                same_event_cluster=excluded.same_event_cluster,
                lexical_score=excluded.lexical_score,
                semantic_score=excluded.semantic_score,
                rank_score=excluded.rank_score,
                reranker_score=excluded.reranker_score,
                lexical_rank=excluded.lexical_rank,
                semantic_rank=excluded.semantic_rank,
                retrieval_methods=excluded.retrieval_methods,
                shared_entity_count=excluded.shared_entity_count,
                shared_entities=excluded.shared_entities,
                semantic_entity_weight=excluded.semantic_entity_weight,
                temporal_gap_days=excluded.temporal_gap_days,
                temporal_decay_factor=excluded.temporal_decay_factor,
                is_current=1
        """, links)

    conn.commit()

    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    claim_columns = [
        "claim_id", "origin_article_id", "origin_passage_id", "claim_text",
        "extraction_method", "extraction_status", "candidate_priority",
        "origin_title", "origin_url", "origin_domain", "origin_published_utc",
        "cluster_id", "entity_names",
    ]
    link_columns = [
        "claim_id", "passage_id", "retrieval_rank", "similarity_score",
        "lexical_score", "semantic_score", "rank_score", "reranker_score", "lexical_rank",
        "semantic_rank", "retrieval_methods", "shared_entity_count",
        "shared_entities", "semantic_entity_weight", "temporal_gap_days",
        "temporal_decay_factor", "same_event_cluster", "assessment",
        "review_notes", "evidence_article_id", "evidence_title",
        "evidence_url", "evidence_domain", "evidence_domain_key",
        "evidence_source_name", "evidence_published_utc", "evidence_text",
    ]
    claims_frame = pd.DataFrame(claims, columns=claim_columns)
    links_frame = pd.DataFrame(links, columns=link_columns)
    _atomic_write_csv(claims_frame, reports / "claim_candidates.csv")
    _atomic_write_csv(links_frame, reports / "claim_evidence_review.csv")

    # Create a reproducible manual-review sample. Labels are intentionally blank.
    sample_path = reports / "retrieval_manual_review_sample.csv"
    if review_sample_claims > 0 and not links_frame.empty:
        unique_ids = sorted(links_frame["claim_id"].dropna().unique().tolist())
        rng = np.random.default_rng(sample_seed)
        sample_n = min(review_sample_claims, len(unique_ids))
        sampled_ids = set(
            rng.choice(unique_ids, size=sample_n, replace=False).tolist()
        )
        sample = links_frame[links_frame["claim_id"].isin(sampled_ids)].copy()
        sample["relevance_label"] = ""
        sample["stance_label"] = ""
        sample["reviewer_notes"] = ""
        _atomic_write_csv(sample, sample_path)
    elif sample_path.exists():
        # Avoid leaving a stale review sample from an earlier run.
        sample_path.unlink()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="Deprecated shared threshold; if set, overrides both component thresholds."
    )
    parser.add_argument("--min-lexical-score", type=float, default=DEFAULT_MIN_LEXICAL_SCORE)
    parser.add_argument("--min-semantic-score", type=float, default=DEFAULT_MIN_SEMANTIC_SCORE)
    parser.add_argument("--max-claims-per-article", type=int, default=DEFAULT_MAX_CLAIMS_PER_ARTICLE)
    parser.add_argument("--max-passages-per-article", type=int, default=DEFAULT_MAX_PASSAGES_PER_ARTICLE)
    parser.add_argument("--max-per-article", type=int, default=1)
    parser.add_argument("--max-per-domain", type=int, default=2)
    parser.add_argument("--retriever", choices=("hybrid", "tfidf"), default="hybrid")
    parser.add_argument("--semantic", action="store_true",
                        help="Backward-compatible alias for --retriever hybrid.")
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--same-cluster-bonus", type=float, default=0.0005)
    parser.add_argument("--entity-one-weight", type=float, default=0.10,
                        help="Semantic RRF multiplier when exactly one entity matches.")
    parser.add_argument("--entity-missing-weight", type=float, default=0.50,
                        help="Semantic multiplier when either side has no extracted entities.")
    parser.add_argument("--temporal-half-life-days", type=float, default=30.0,
                        help="Publication-date gap at which the ranking multiplier is 0.5.")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Pretrained cross-encoder used to rerank hybrid retrieval candidates.")
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--review-sample-claims", type=int, default=DEFAULT_REVIEW_SAMPLE_CLAIMS,
                        help="Number of claims sampled for manual review; 0 disables sample.")
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.min_similarity is not None:
        if not 0 <= args.min_similarity <= 1:
            parser.error("--min-similarity must be between 0 and 1.")
        args.min_lexical_score = args.min_similarity
        args.min_semantic_score = args.min_similarity
    if not 0 <= args.min_lexical_score <= 1:
        parser.error("--min-lexical-score must be between 0 and 1.")
    if not -1 <= args.min_semantic_score <= 1:
        parser.error("--min-semantic-score must be between -1 and 1.")
    if min(
        args.max_claims_per_article, args.max_passages_per_article,
        args.max_per_article, args.max_per_domain, args.batch_size
    ) < 1:
        parser.error("Article/domain/batch limits must be >= 1.")
    if args.candidate_pool < args.top_k:
        parser.error("--candidate-pool must be >= --top-k.")
    if args.review_sample_claims < 0:
        parser.error("--review-sample-claims cannot be negative.")
    if args.same_cluster_bonus < 0 or args.same_cluster_bonus > 0.01:
        parser.error("--same-cluster-bonus must be between 0 and 0.01.")
    if not 0.0 <= args.entity_one_weight <= 1.0:
        parser.error("--entity-one-weight must be between 0 and 1.")
    if not 0.0 <= args.entity_missing_weight <= 1.0:
        parser.error("--entity-missing-weight must be between 0 and 1.")
    if args.temporal_half_life_days <= 0:
        parser.error("--temporal-half-life-days must be > 0.")
    if not args.db.exists():
        raise FileNotFoundError(args.db)

    with sqlite3.connect(args.db) as conn:
        articles = load_articles(conn)
        if articles.empty:
            raise RuntimeError("No articles with extracted body text found.")
        claims, passages = create_records(
            articles, args.max_claims_per_article, args.max_passages_per_article
        )
        if not claims:
            raise RuntimeError("No claim candidates were extracted.")
        retriever = "hybrid" if args.semantic else args.retriever
        links = retrieve(
            claims, passages, args.top_k, args.min_similarity,
            retriever=retriever,
            model_name=args.semantic_model,
            max_per_domain=args.max_per_domain,
            max_per_article=args.max_per_article,
            candidate_pool=args.candidate_pool,
            batch_size=args.batch_size,
            same_cluster_bonus=args.same_cluster_bonus,
            min_lexical_score=args.min_lexical_score,
            min_semantic_score=args.min_semantic_score,
            entity_one_weight=args.entity_one_weight,
            entity_missing_weight=args.entity_missing_weight,
            temporal_half_life_days=args.temporal_half_life_days,
            reranker_model=args.reranker_model,
            rerank_batch_size=args.rerank_batch_size,
        )
        persist(
            conn, claims, passages, links, args.reports,
            review_sample_claims=args.review_sample_claims,
            sample_seed=args.sample_seed,
        )

    print("\n=== CLAIM / EVIDENCE RETRIEVAL SUMMARY ===")
    print(f"Articles with body text: {len(articles)}")
    print(f"Passages indexed: {len(passages)}")
    print(f"Claim candidates: {len(claims)}")
    print(f"Claim/evidence links: {len(links)}")
    print(f"Retriever: {'hybrid TF-IDF + semantic (RRF)' if retriever == 'hybrid' else 'TF-IDF'}")
    print(f"Cross-encoder reranker: {args.reranker_model}")
    print(f"Candidate pool per method: {args.candidate_pool}")
    print(f"Maximum passages per article: {args.max_per_article}")
    print(f"Maximum articles per domain: {args.max_per_domain}")
    print("Cross-cluster evidence remains eligible.")
    print("Semantic contribution is conditioned on sentence-grounded entity overlap; lexical retrieval remains independent.")
    print(f"Semantic multiplier for exactly one shared entity: {args.entity_one_weight:.2f}")
    print(f"Semantic multiplier when entity extraction is missing: {args.entity_missing_weight:.2f}")
    print(f"Publication-date decay half-life: {args.temporal_half_life_days:g} days")
    print("Publication dates are proxies, not event dates; decay is ranking-only.")
    print("Automatic support/contradiction decisions: 0")
    print("Existing human assessments and review notes are preserved.")
    print(f"Claim candidates: {args.reports / 'claim_candidates.csv'}")
    print(f"Review queue: {args.reports / 'claim_evidence_review.csv'}")
    if args.review_sample_claims:
        print(f"Manual review sample: {args.reports / 'retrieval_manual_review_sample.csv'}")


if __name__ == "__main__":
    main()
