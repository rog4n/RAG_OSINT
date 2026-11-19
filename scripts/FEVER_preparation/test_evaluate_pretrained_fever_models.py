import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("evaluate_pretrained_fever_models.py")
spec = importlib.util.spec_from_file_location("fever_pretrained_eval", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestPretrainedFeverEvaluation(unittest.TestCase):
    def test_read_jsonl_schema(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "validation_pairs.jsonl"
            rows = [{
                "claim_text": "Claim",
                "passage_text": "Evidence",
                "relevance_label": "relevant",
                "label_origin": mod.GOLD_ORIGIN,
                "stance_label": "supports",
            }]
            p.write_text("".join(json.dumps(x) + "\n" for x in rows))
            self.assertEqual(len(mod.read_jsonl(p, "validation")), 1)

    def test_relevance_metrics_separates_origin(self):
        rows = [
            {"relevance_label": "relevant", "label_origin": mod.GOLD_ORIGIN},
            {"relevance_label": "irrelevant", "label_origin": mod.SYNTHETIC_ORIGIN},
            {"relevance_label": "relevant", "label_origin": mod.GOLD_ORIGIN},
            {"relevance_label": "irrelevant", "label_origin": mod.SYNTHETIC_ORIGIN},
        ]
        result = mod.relevance_metrics(rows, np.array([2.0, -1.0, 1.0, -2.0]))
        self.assertEqual(result["ranking_metrics"]["roc_auc"], 1.0)
        self.assertEqual(result["gold_evidence_score_diagnostics"]["count"], 2)
        self.assertEqual(result["synthetic_negative_score_diagnostics"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
