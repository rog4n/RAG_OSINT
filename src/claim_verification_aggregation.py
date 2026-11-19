#!/usr/bin/env python3
"""Aggregate claim/evidence records into auditable dossiers and conservative claim statuses.

NLI predictions are experimental triage signals only. Automated support/contradiction
verdicts are disabled. A final support/contradiction status requires explicit reviewer
assessments on at least two distinct publisher domains.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "new_articles.db"
DEFAULT_REPORTS = ROOT / "reports"

HUMAN_SUPPORT = {"supports", "support", "supported", "human_support"}
HUMAN_CONTRADICTION = {"contradicts", "contradict", "contradicted", "human_contradiction"}
HUMAN_IRRELEVANT = {"irrelevant", "not_relevant", "unrelated"}
HUMAN_INSUFFICIENT = {"insufficient", "uncertain", "inconclusive", "unassessed", ""}

SUMMARY_COLUMNS = [
    "claim_id", "claim_text", "claim_url", "candidate_evidence_count",
    "relevance_gate_pass_count", "distinct_evidence_articles",
    "distinct_publisher_domains", "human_support_count", "human_support_domains",
    "human_contradiction_count", "human_contradiction_domains",
    "human_irrelevant_count", "nli_support_signals", "nli_contradiction_signals",
    "claim_status", "automated_verdict_enabled", "status_note",
]

def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()

def domain_key(url, domain, source):
    candidate = str(url or "").strip()
    if candidate:
        try:
            host = urlparse(candidate if "://" in candidate else "//" + candidate).hostname
            if host:
                return host.lower().removeprefix("www.")
        except ValueError:
            pass
    d = str(domain or "").strip().lower().removeprefix("www.")
    if d:
        return d
    return "source:" + norm(source) if norm(source) else "unknown"

def load_dossier(conn):
    """Load one row per current claim/evidence pair, retaining current claims with no hits."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"claim_candidates", "evidence_passages", "claim_evidence"}
    if not required.issubset(tables):
        raise RuntimeError("Missing retrieval tables. Run evidence retrieval first.")

    nli_exists = "claim_evidence_nli" in tables
    article_cols = {r[1] for r in conn.execute('PRAGMA table_info("articles")')} if "articles" in tables else set()
    ce_cols = {r[1] for r in conn.execute('PRAGMA table_info("claim_evidence")')}
    cc_cols = {r[1] for r in conn.execute('PRAGMA table_info("claim_candidates")')}
    nli_cols = {r[1] for r in conn.execute('PRAGMA table_info("claim_evidence_nli")')} if nli_exists else set()
    missing_current = []
    if "is_current" not in cc_cols:
        missing_current.append("claim_candidates.is_current")
    if "is_current" not in ce_cols:
        missing_current.append("claim_evidence.is_current")
    if missing_current:
        raise RuntimeError(
            "Current-run tracking is not initialized (missing "
            + ", ".join(missing_current)
            + "). Run the updated evidence retrieval stage before aggregation."
        )

    def opt(cols, alias, col, fallback="''"):
        return f"{alias}.{col}" if col in cols else fallback

    url_expr = opt(article_cols, "a", "url")
    domain_expr = opt(article_cols, "a", "domain")
    source_expr = opt(article_cols, "a", "source_name", opt(article_cols, "a", "source"))
    evidence_date = opt(article_cols, "a", "published_utc")
    origin_url = opt(cc_cols, "cc", "origin_url")
    origin_domain = opt(cc_cols, "cc", "origin_domain")
    origin_date = opt(cc_cols, "cc", "origin_published_utc")
    extraction_method = opt(cc_cols, "cc", "extraction_method")
    extraction_status = opt(cc_cols, "cc", "extraction_status")
    assessment = opt(ce_cols, "ce", "assessment", "'unassessed'")
    notes = opt(ce_cols, "ce", "review_notes")

    # A candidate with no current retrieval hits must remain in the dossier,
    # but a candidate whose only links are stale must not leak into this run.
    current_ce_on = "AND COALESCE(ce.is_current,1)=1" if "is_current" in ce_cols else ""
    current_cc_where = "AND COALESCE(cc.is_current,1)=1" if "is_current" in cc_cols else ""

    if nli_exists:
        # Multiple model runs are allowed in the NLI table. For a given pair,
        # use the most recently scored model row so the dossier does not
        # multiply evidence rows or inflate stance counts.
        if {"scored_utc", "model_name"}.issubset(nli_cols):
            nli_source = """(
                SELECT * FROM (
                    SELECT n0.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY claim_id, passage_id
                               ORDER BY COALESCE(scored_utc,'') DESC, model_name DESC
                           ) AS _rownum
                    FROM claim_evidence_nli n0
                ) WHERE _rownum=1
            )"""
        else:
            nli_source = "claim_evidence_nli"
        nli_join = f"LEFT JOIN {nli_source} n ON n.claim_id=ce.claim_id AND n.passage_id=ce.passage_id"
        nli_select = f"""{opt(nli_cols,'n','predicted_stance')} AS nli_stance,
            {opt(nli_cols,'n','raw_top_stance')} AS nli_raw_stance,
            {opt(nli_cols,'n','status')} AS nli_status,
            {opt(nli_cols,'n','confidence','NULL')} AS nli_confidence,
            {opt(nli_cols,'n','relevance_status')} AS relevance_status"""
    else:
        nli_join = ""
        nli_select = "'' AS nli_stance, '' AS nli_raw_stance, '' AS nli_status, NULL AS nli_confidence, '' AS relevance_status"

    article_join = "LEFT JOIN articles a ON a.article_id=ep.article_id" if "articles" in tables else ""
    query = f"""
      SELECT cc.claim_id, cc.claim_text, {origin_url} AS claim_url,
             {origin_domain} AS claim_domain, {origin_date} AS claim_published_utc,
             {extraction_method} AS extraction_method,
             {extraction_status} AS extraction_status,
             ce.passage_id, ce.retrieval_rank, {assessment} AS human_assessment,
             {notes} AS review_notes, ep.article_id AS evidence_article_id,
             ep.passage_text AS evidence_text, {url_expr} AS evidence_url,
             {domain_expr} AS evidence_domain, {source_expr} AS evidence_source,
             {evidence_date} AS evidence_published_utc,
             {nli_select}
      FROM claim_candidates cc
      LEFT JOIN claim_evidence ce
        ON ce.claim_id=cc.claim_id {current_ce_on}
      LEFT JOIN evidence_passages ep ON ep.passage_id=ce.passage_id
      {article_join}
      {nli_join}
      WHERE 1=1 {current_cc_where}
      ORDER BY cc.claim_id, ce.retrieval_rank
    """
    return pd.read_sql_query(query, conn)

def build_summaries(dossier):
    if dossier.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS), dossier
    dossier = dossier.copy()
    dossier["domain_key"] = dossier.apply(
        lambda r: domain_key(r.get("evidence_url"), r.get("evidence_domain"), r.get("evidence_source")), axis=1
    )
    dossier["assessment_norm"] = dossier["human_assessment"].map(norm)
    dossier["is_human_support"] = dossier["assessment_norm"].isin(HUMAN_SUPPORT)
    dossier["is_human_contradiction"] = dossier["assessment_norm"].isin(HUMAN_CONTRADICTION)
    dossier["is_human_irrelevant"] = dossier["assessment_norm"].isin(HUMAN_IRRELEVANT)
    dossier["is_human_assessed"] = (
        dossier["is_human_support"] | dossier["is_human_contradiction"] |
        dossier["is_human_irrelevant"] |
        (~dossier["assessment_norm"].isin(HUMAN_INSUFFICIENT))
    )
    rows = []
    for claim_id, group in dossier.groupby("claim_id", sort=True):
        evidence_rows = group[group["passage_id"].notna()].copy()
        supports = evidence_rows[evidence_rows.is_human_support]
        contradicts = evidence_rows[evidence_rows.is_human_contradiction]
        support_domains = set(supports.domain_key)
        contradiction_domains = set(contradicts.domain_key)
        # Count only emitted NLI signals on pairs that passed the relevance gate.
        # Abstentions and skipped pairs are not stance signals.
        signal_mask = (
            (group.nli_status == "model_signal_only")
            & (group.relevance_status == "likely_relevant")
        )
        nli_support = int((signal_mask & (group.nli_stance == "supports")).sum())
        nli_contradict = int((signal_mask & (group.nli_stance == "contradicts")).sum())
        relevance_passed = int((evidence_rows.relevance_status == "likely_relevant").sum())
        # NLI predictions are never allowed to set the final claim status.
        if support_domains and contradiction_domains:
            status = "mixed_human_evidence_review"
        elif len(support_domains) >= 2:
            status = "human_reviewed_support"
        elif len(contradiction_domains) >= 2:
            status = "human_reviewed_contradiction"
        elif supports.shape[0] or contradicts.shape[0]:
            status = "human_review_incomplete_single_source"
        elif group.is_human_irrelevant.any():
            status = "human_reviewed_no_relevant_evidence"
        else:
            status = "unverified_model_triage_only"
        first = group.iloc[0]
        rows.append({
            "claim_id": claim_id,
            "claim_text": first.claim_text,
            "claim_url": first.claim_url,
            "candidate_evidence_count": int(len(evidence_rows)),
            "relevance_gate_pass_count": relevance_passed,
            "distinct_evidence_articles": int(evidence_rows.evidence_article_id.nunique()),
            "distinct_publisher_domains": int(evidence_rows.domain_key.nunique()),
            "human_support_count": int(len(supports)),
            "human_support_domains": len(support_domains),
            "human_contradiction_count": int(len(contradicts)),
            "human_contradiction_domains": len(contradiction_domains),
            "human_irrelevant_count": int(evidence_rows.is_human_irrelevant.sum()),
            "nli_support_signals": nli_support,
            "nli_contradiction_signals": nli_contradict,
            "claim_status": status,
            "automated_verdict_enabled": 0,
            "status_note": (
                "NLI outputs are not used as truth labels. Two distinct domains are a minimum "
                "review-evidence threshold, not proof of source independence."
            )
        })
    return pd.DataFrame(rows), dossier

def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = ap.parse_args()
    if not args.db.exists():
        ap.error(f"Database not found: {args.db}")
    with sqlite3.connect(args.db) as conn:
        dossier = load_dossier(conn)
        summaries, dossier = build_summaries(dossier)
        conn.execute("""
          CREATE TABLE IF NOT EXISTS claim_verification_summary (
            claim_id TEXT PRIMARY KEY, claim_status TEXT NOT NULL,
            candidate_evidence_count INTEGER NOT NULL,
            relevance_gate_pass_count INTEGER NOT NULL,
            distinct_evidence_articles INTEGER NOT NULL,
            distinct_publisher_domains INTEGER NOT NULL,
            human_support_count INTEGER NOT NULL,
            human_support_domains INTEGER NOT NULL,
            human_contradiction_count INTEGER NOT NULL,
            human_contradiction_domains INTEGER NOT NULL,
            nli_support_signals INTEGER NOT NULL,
            nli_contradiction_signals INTEGER NOT NULL,
            automated_verdict_enabled INTEGER NOT NULL DEFAULT 0,
            generated_utc TEXT DEFAULT CURRENT_TIMESTAMP,
            is_current INTEGER NOT NULL DEFAULT 1
          )
        """)
        summary_cols = {r[1] for r in conn.execute('PRAGMA table_info("claim_verification_summary")')}
        if "is_current" not in summary_cols:
            conn.execute("ALTER TABLE claim_verification_summary ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1")
        conn.execute("UPDATE claim_verification_summary SET is_current=0")
        if not summaries.empty:
            rows = summaries.to_dict("records")
            conn.executemany("""
              INSERT INTO claim_verification_summary (
                claim_id,claim_status,candidate_evidence_count,relevance_gate_pass_count,
                distinct_evidence_articles,distinct_publisher_domains,human_support_count,
                human_support_domains,human_contradiction_count,human_contradiction_domains,
                nli_support_signals,nli_contradiction_signals,automated_verdict_enabled,is_current
              ) VALUES (
                :claim_id,:claim_status,:candidate_evidence_count,:relevance_gate_pass_count,
                :distinct_evidence_articles,:distinct_publisher_domains,:human_support_count,
                :human_support_domains,:human_contradiction_count,:human_contradiction_domains,
                :nli_support_signals,:nli_contradiction_signals,0,1
              ) ON CONFLICT(claim_id) DO UPDATE SET
                claim_status=excluded.claim_status,
                candidate_evidence_count=excluded.candidate_evidence_count,
                relevance_gate_pass_count=excluded.relevance_gate_pass_count,
                distinct_evidence_articles=excluded.distinct_evidence_articles,
                distinct_publisher_domains=excluded.distinct_publisher_domains,
                human_support_count=excluded.human_support_count,
                human_support_domains=excluded.human_support_domains,
                human_contradiction_count=excluded.human_contradiction_count,
                human_contradiction_domains=excluded.human_contradiction_domains,
                nli_support_signals=excluded.nli_support_signals,
                nli_contradiction_signals=excluded.nli_contradiction_signals,
                automated_verdict_enabled=0,generated_utc=CURRENT_TIMESTAMP,is_current=1
            """, rows)
        conn.commit()
    args.reports.mkdir(parents=True, exist_ok=True)
    atomic_csv(dossier, args.reports / "claim_evidence_dossier.csv")
    atomic_csv(summaries, args.reports / "claim_verification_summary.csv")
    print(f"Claims summarized: {len(summaries)}")
    print(f"Evidence rows: {len(dossier)}")
    if not summaries.empty:
        print(summaries.claim_status.value_counts().to_string())
    print("Automated factual verdicts: DISABLED")
    print(f"Dossier: {args.reports / 'claim_evidence_dossier.csv'}")
    print(f"Summary: {args.reports / 'claim_verification_summary.csv'}")

if __name__ == "__main__":
    main()
