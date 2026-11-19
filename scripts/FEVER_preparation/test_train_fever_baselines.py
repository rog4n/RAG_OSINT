import json
import tempfile
import unittest
from pathlib import Path

from train_fever_baselines import TrainingDataError, read_jsonl, train


def build_rows(split):
    rows = []
    examples = [
        ("supports", "The capital of France is Paris", "Paris is the capital city of France", "France capital Paris"),
        ("supports", "Water freezes at zero degrees Celsius", "At 0 degrees Celsius, water freezes", "water freezes Celsius"),
        ("contradicts", "The capital of France is Berlin", "Paris is the capital city of France", "France capital Paris"),
        ("contradicts", "Water boils at zero degrees Celsius", "Water boils at 100 degrees Celsius", "water boils Celsius"),
    ]
    for index, (stance, claim, passage, unrelated) in enumerate(examples):
        claim_id = f"{split}-claim-{index}"
        rows.append({
            "pair_id": f"{split}-gold-{index}", "claim_id": claim_id,
            "claim_text": claim, "passage_text": passage,
            "relevance_label": "relevant", "stance_label": stance,
            "claim_level_label": "SUPPORTS" if stance == "supports" else "REFUTES",
            "split": split, "label_origin": "fever_gold_evidence",
            "is_synthetic_label": False,
        })
        rows.append({
            "pair_id": f"{split}-negative-{index}", "claim_id": claim_id,
            "claim_text": claim, "passage_text": unrelated,
            "relevance_label": "irrelevant", "stance_label": None,
            "claim_level_label": "SUPPORTS" if stance == "supports" else "REFUTES",
            "split": split, "label_origin": "synthetic_random_negative",
            "is_synthetic_label": True,
        })
    return rows


class TestFeverBaselineTraining(unittest.TestCase):
    def test_rejects_synthetic_negative_with_stance_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            row = build_rows("train")[1]
            row["stance_label"] = "contradicts"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(TrainingDataError):
                read_jsonl(path, "train")

    def test_trains_both_separate_baselines_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "pairs"
            output_dir = root / "models"
            data_dir.mkdir()
            for split, filename in (("train", "train_pairs.jsonl"), ("validation", "validation_pairs.jsonl")):
                with (data_dir / filename).open("w", encoding="utf-8") as handle:
                    for row in build_rows(split):
                        handle.write(json.dumps(row) + "\n")
            metrics = train(data_dir, output_dir, max_features=1000, min_df=1, c_value=1.0)
            self.assertTrue((output_dir / "tfidf_vectorizer.joblib").is_file())
            self.assertTrue((output_dir / "relevance_linear_svc.joblib").is_file())
            self.assertTrue((output_dir / "stance_linear_svc.joblib").is_file())
            self.assertTrue((output_dir / "validation_metrics.json").is_file())
            self.assertEqual(metrics["split_counts"]["claim_id_overlap"], 0)
            self.assertEqual(metrics["split_counts"]["train_gold_stance_labels"], {"supports": 2, "contradicts": 2})
            self.assertEqual(metrics["split_counts"]["validation_gold_stance_labels"], {"supports": 2, "contradicts": 2})
            self.assertEqual(metrics["relevance_validation"]["synthetic_negative_only"]["synthetic_pair_count"], 4)
            self.assertEqual(metrics["stance_validation_on_gold_evidence_only"]["label_order"], ["contradicts", "supports"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
