import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

nli = load_module("nli_under_test", ROOT / "src" / "nli_verification.py")
agg = load_module("agg_under_test", ROOT / "src" / "claim_verification_aggregation.py")

class FakeTokenizer:
    def __call__(self, premises, hypotheses, **kwargs):
        return {"input_ids": torch.ones((len(premises), 4), dtype=torch.long)}

class FakeConfig:
    _commit_hash = "synthetic-test"

class FakeModel:
    config = FakeConfig()
    def __call__(self, **kwargs):
        return type("Output", (), {
            "logits": torch.tensor([[3.0, 0.1, 0.1]] * kwargs["input_ids"].shape[0])
        })()

class VerificationTests(unittest.TestCase):
    def test_relevance_gate_relevant(self):
        r = nli.relevance_gate(
            "Rwanda's military supports M23 in eastern Congo.",
            "Rwanda deployed troops in support of M23 in Congo.",
            "Rwanda; M23"
        )
        self.assertEqual(r["relevance_status"], "likely_relevant")

    def test_relevance_gate_unrelated(self):
        r = nli.relevance_gate(
            "The Greenville Police concluded an investigation into a death.",
            "A startup aims to make its product affordable."
        )
        self.assertEqual(r["relevance_status"], "likely_unrelated")

    def test_nli_only_scores_gate_passed_pairs(self):
        frame = pd.DataFrame([
            {"claim_id":"c1","passage_id":"p1",
             "claim_text":"Rwanda's military supports M23 in eastern Congo.",
             "evidence_text":"Rwanda deployed troops in support of M23 in Congo.",
             "origin_article_id":"a1","evidence_article_id":"a2",
             "retrieval_rank":1,"stored_shared_entities":"Rwanda; M23"},
            {"claim_id":"c2","passage_id":"p2",
             "claim_text":"Greenville Police concluded an investigation into a death.",
             "evidence_text":"A startup aims to make its product affordable.",
             "origin_article_id":"a3","evidence_article_id":"a4",
             "retrieval_rank":1,"stored_shared_entities":""},
        ])
        result = nli.score_pairs(
            frame, FakeTokenizer(), FakeModel(),
            {0:"supports",1:"contradicts",2:"insufficient"},
            "cpu", torch, "fake-model"
        )
        unrelated = result[result.claim_id == "c2"].iloc[0]
        self.assertEqual(unrelated.predicted_stance, "abstain")
        self.assertEqual(unrelated.status, "skipped_likely_unrelated")
        self.assertTrue((result.verdict_eligible == 0).all())

    def test_persistence_and_migration(self):
        frame = pd.DataFrame([{
            "pair_id":"pair1","claim_id":"c1","passage_id":"p1",
            "model_name":"fake","model_version":"v1","status":"model_signal_only",
            "predicted_stance":"supports","raw_top_stance":"supports",
            "p_supports":0.9,"p_contradicts":0.05,"p_insufficient":0.05,
            "confidence":0.9,"margin":0.85,"same_article":0,
            "scored_utc":"2026-11-01T00:00:00+00:00",
            "relevance_status":"likely_relevant","relevance_shared_token_count":3,
            "relevance_shared_tokens":"rwanda; m23; congo",
            "relevance_token_overlap_ratio":0.5,"relevance_shared_entity_count":1,
            "relevance_shared_entities":"rwanda","relevance_number_mismatch_flag":0,
            "verdict_eligible":0,"interpretation":"triage only"
        }])
        conn = sqlite3.connect(":memory:")
        # Simulate the older NLI table schema to test migrations.
        conn.execute("""
          CREATE TABLE claim_evidence_nli (
            pair_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL, passage_id TEXT NOT NULL,
            model_name TEXT NOT NULL, model_version TEXT, status TEXT NOT NULL,
            predicted_stance TEXT NOT NULL, raw_top_stance TEXT, p_supports REAL,
            p_contradicts REAL, p_insufficient REAL, confidence REAL, margin REAL,
            same_article INTEGER NOT NULL DEFAULT 0, scored_utc TEXT NOT NULL,
            UNIQUE(claim_id,passage_id,model_name)
          )
        """)
        nli.persist_results(conn, frame)
        row = conn.execute(
            "SELECT relevance_status,verdict_eligible FROM claim_evidence_nli"
        ).fetchone()
        self.assertEqual(row, ("likely_relevant", 0))

    def test_aggregation_never_uses_nli_as_verdict(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
          CREATE TABLE claim_candidates(claim_id TEXT, claim_text TEXT, origin_url TEXT, origin_domain TEXT);
          CREATE TABLE evidence_passages(passage_id TEXT, article_id TEXT, passage_text TEXT);
          CREATE TABLE claim_evidence(claim_id TEXT, passage_id TEXT, retrieval_rank INTEGER, assessment TEXT, review_notes TEXT, is_current INTEGER);
          CREATE TABLE articles(article_id TEXT, url TEXT, domain TEXT, source_name TEXT);
          CREATE TABLE claim_evidence_nli(claim_id TEXT, passage_id TEXT, predicted_stance TEXT,
            raw_top_stance TEXT, status TEXT, confidence REAL, relevance_status TEXT);
        """)
        conn.execute("INSERT INTO claim_candidates VALUES ('c1','A claim','','')")
        conn.execute("INSERT INTO evidence_passages VALUES ('p1','a1','evidence')")
        conn.execute("INSERT INTO claim_evidence VALUES ('c1','p1',1,'unassessed','',1)")
        conn.execute("INSERT INTO articles VALUES ('a1','https://source.example/a','source.example','Source')")
        conn.execute("INSERT INTO claim_evidence_nli VALUES ('c1','p1','contradicts','contradicts','model_signal_only',0.99,'likely_relevant')")
        dossier = agg.load_dossier(conn)
        summary, _ = agg.build_summaries(dossier)
        self.assertEqual(summary.iloc[0].claim_status, "unverified_model_triage_only")
        self.assertEqual(summary.iloc[0].automated_verdict_enabled, 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
