import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.nli_verification import (
    NLIThresholds, load_pairs, persist_results, resolve_label_map,
    score_pairs, stable_pair_id,
)


class FakeConfig:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class FakeModel:
    config = FakeConfig()
    def __call__(self, **kwargs):
        class Output: pass
        out = Output()
        # Rows deliberately return confident support, contradiction, neutral.
        out.logits = FakeTensor([[0.1, 3.0, 0.1], [3.0, 0.1, 0.1], [0.1, 0.1, 3.0]])
        return out


class FakeTensor:
    def __init__(self, values): self.values = np.asarray(values, dtype=float)
    def to(self, device): return self
    def float(self): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


class FakeTorch:
    class inference_mode:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    @staticmethod
    def softmax(tensor, dim=-1):
        x = tensor.values
        e = np.exp(x - x.max(axis=dim, keepdims=True))
        return FakeTensor(e / e.sum(axis=dim, keepdims=True))


class FakeTokenizer:
    def __call__(self, premises, hypotheses, **kwargs):
        return {"dummy": FakeTensor([[0]] * len(premises))}


class FakeModelBatch(FakeModel):
    def __call__(self, **kwargs):
        class Output: pass
        out = Output()
        out.logits = FakeTensor([[0.1, 3.0, 0.1]] * 3)
        return out


class NliTests(unittest.TestCase):
    def test_label_map_refuses_ambiguous(self):
        class Bad: id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        with self.assertRaises(RuntimeError): resolve_label_map(Bad())

    def test_pair_id_stable(self):
        self.assertEqual(stable_pair_id("c", "p", "m"), stable_pair_id("c", "p", "m"))
        self.assertNotEqual(stable_pair_id("c", "p", "m"), stable_pair_id("c", "p2", "m"))

    def test_load_pairs_and_persist_preserves_human_review(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript('''
        CREATE TABLE claim_candidates(claim_id TEXT PRIMARY KEY, claim_text TEXT, origin_article_id TEXT, origin_title TEXT, origin_url TEXT, origin_domain TEXT, origin_published_utc TEXT);
        CREATE TABLE evidence_passages(passage_id TEXT PRIMARY KEY, article_id TEXT, passage_text TEXT);
        CREATE TABLE claim_evidence(claim_id TEXT, passage_id TEXT, retrieval_rank INTEGER, assessment TEXT, same_event_cluster INTEGER, similarity_score REAL, is_current INTEGER DEFAULT 1, PRIMARY KEY(claim_id,passage_id));
        INSERT INTO claim_candidates VALUES('c','The city opened the bridge.','a1','title','url','domain','2026-01-01');
        INSERT INTO evidence_passages VALUES('p','a2','The bridge was opened by the city.');
        INSERT INTO claim_evidence VALUES('c','p',1,'supports',0,0.8,1);
        ''')
        frame = load_pairs(conn)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].claim_text, 'The city opened the bridge.')
        result = pd.DataFrame([{
            'pair_id':'pair','claim_id':'c','passage_id':'p','model_name':'m','model_version':'v',
            'status':'predicted','predicted_stance':'supports','raw_top_stance':'supports',
            'p_supports':0.9,'p_contradicts':0.05,'p_insufficient':0.05,'confidence':0.9,
            'margin':0.85,'same_article':0,'scored_utc':'2026-01-01T00:00:00+00:00'
        }])
        persist_results(conn, result)
        persist_results(conn, result)
        self.assertEqual(conn.execute("SELECT assessment FROM claim_evidence").fetchone()[0], 'supports')
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM claim_evidence_nli").fetchone()[0], 1)
        conn.close()

    def test_score_abstains_same_article_and_maps_labels(self):
        frame = pd.DataFrame([
            {'claim_id':'c1','passage_id':'p1','claim_text':'Claim A','evidence_text':'Evidence A','origin_article_id':'a','evidence_article_id':'b'},
            {'claim_id':'c2','passage_id':'p2','claim_text':'Claim B','evidence_text':'Evidence B','origin_article_id':'a','evidence_article_id':'a'},
            {'claim_id':'c3','passage_id':'p3','claim_text':'Claim C','evidence_text':'Evidence C','origin_article_id':'a','evidence_article_id':'c'},
        ])
        results = score_pairs(frame, FakeTokenizer(), FakeModel(), resolve_label_map(FakeConfig()),
                              'cpu', FakeTorch(), 'fake-model', batch_size=3,
                              thresholds=NLIThresholds(0.5, 0.1, 0.5, 0.5))
        self.assertEqual(results.iloc[0].predicted_stance, 'supports')
        self.assertEqual(results.iloc[1].status, 'abstained_same_article')
        self.assertEqual(results.iloc[1].predicted_stance, 'abstain')


if __name__ == '__main__': unittest.main()
