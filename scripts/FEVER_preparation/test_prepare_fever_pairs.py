from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_fever_pairs import validate_split

from prepare_fever_pairs import (
    DatasetError,
    content_tokens,
    parse_evidence_groups,
    parse_wiki_lines,
    prepare_dataset,
    read_claim_file,
)


class FEVERParsingTests(unittest.TestCase):
    def test_parse_wiki_lines_uses_sentence_column_only(self):
        raw = "0\tFirst sentence.\tEntity\tEntity_Page\n1\tSecond sentence.\n2\tThird sentence.\tOther"
        self.assertEqual(
            list(parse_wiki_lines(raw)),
            [(0, "First sentence."), (1, "Second sentence."), (2, "Third sentence.")],
        )

    def test_parse_wiki_lines_filters_requested_ids(self):
        raw = "0\tFirst sentence.\n1\tSecond sentence."
        self.assertEqual(list(parse_wiki_lines(raw, {1})), [(1, "Second sentence.")])

    def test_parse_evidence_ignores_null_nei_tuple(self):
        refs = parse_evidence_groups([[[10, None, None, None]]], "row")
        self.assertEqual(refs, ())

    def test_parse_evidence_preserves_alternative_group_indices(self):
        raw = [
            [[10, 20, "Page_A", 3], [10, 21, "Page_A", 4]],
            [[11, 22, "Page_B", 1]],
        ]
        refs = parse_evidence_groups(raw, "row")
        self.assertEqual(
            [(r.page_id, r.sentence_id, r.group_index) for r in refs],
            [("Page_A", 3, 0), ("Page_A", 4, 0), ("Page_B", 1, 1)],
        )

    def test_content_tokens_removes_stopwords_and_normalizes(self):
        self.assertEqual(content_tokens("THE quick fox, and a dog!"), {"quick", "fox", "dog"})


class FEVEREndToEndTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_builds_gold_pairs_and_explicit_synthetic_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fever = root / "fever"
            wiki = root / "wiki" / "wiki-pages"
            out = root / "out"
            wiki.mkdir(parents=True)
            fever.mkdir(parents=True)

            train_rows = [
                {
                    "id": 1,
                    "verifiable": "VERIFIABLE",
                    "label": "SUPPORTS",
                    "claim": "A dragon is a mythological creature from folklore.",
                    "evidence": [[[1, 2, "Gold_Dragon", 0], [1, 5, "Gold_Dragon", 1]], [[2, 2, "Gold_Dragon", 0], [2, 5, "Gold_Dragon", 1]]],
                },
                {
                    "id": 2,
                    "verifiable": "VERIFIABLE",
                    "label": "REFUTES",
                    "claim": "The moon is made of cheese.",
                    "evidence": [[[1, 3, "Gold_Moon", 0]]],
                },
                {
                    "id": 3,
                    "verifiable": "NOT VERIFIABLE",
                    "label": "NOT ENOUGH INFO",
                    "claim": "A lemur can pilot submarines.",
                    "evidence": [[[1, None, None, None]]],
                },
            ]
            dev_rows = [
                {
                    "id": 4,
                    "verifiable": "VERIFIABLE",
                    "label": "SUPPORTS",
                    "claim": "Quartz is a mineral.",
                    "evidence": [[[1, 4, "Gold_Quartz", 0]]],
                }
            ]
            self.write_jsonl(fever / "train.jsonl", train_rows)
            self.write_jsonl(fever / "shared_task_dev.jsonl", dev_rows)

            wiki_rows = [
                {"id": "Gold_Dragon", "text": "", "lines": "0\tA dragon is a mythological creature.\tDragon\tDragon\n1\tThe creature appears in traditional folklore.\tFolklore\tFolklore"},
                {"id": "Gold_Moon", "text": "", "lines": "0\tThe moon is composed primarily of rock.\tMoon\tMoon"},
                {"id": "Gold_Quartz", "text": "", "lines": "0\tQuartz is a mineral found in many rocks.\tQuartz\tQuartz"},
            ]
            # Ensure there are enough non-evidence pages in BOTH deterministic pools.
            for idx in range(40):
                page_id = f"Random_{idx}"
                # Identify the same fixed set of unrelated sentences for every page.
                wiki_rows.append({"id": page_id, "text": "", "lines":
                    f"0\tBasketball players train during season number {idx}.\n"
                    f"1\tAstronomers study distant galaxies and stars number {idx}."})
            self.write_jsonl(wiki / "wiki-000.jsonl", wiki_rows)

            manifest = prepare_dataset(
                fever_dir=fever,
                wiki_dir=wiki,
                output_dir=out,
                negatives_per_claim=1,
                reservoir_size=100,
                seed=7,
                max_negative_attempts=100,
            )

            train_pairs = [json.loads(line) for line in (out / "train_pairs.jsonl").read_text().splitlines()]
            dev_pairs = [json.loads(line) for line in (out / "validation_pairs.jsonl").read_text().splitlines()]
            all_pairs = train_pairs + dev_pairs

            gold = [p for p in all_pairs if p["label_origin"] == "fever_gold_evidence"]
            synthetic = [p for p in all_pairs if p["label_origin"] == "synthetic_random_negative"]
            self.assertEqual(len(gold), 4)
            self.assertEqual(len({p["pair_id"] for p in gold}), 4)
            self.assertEqual({p["stance_label"] for p in gold}, {"supports", "contradicts"})
            self.assertTrue(all(p["relevance_label"] == "relevant" for p in gold))
            dragon_pairs = [p for p in gold if p["claim_id"] == "1"]
            self.assertEqual(len(dragon_pairs), 2)
            dragon_pair = dragon_pairs[0]
            self.assertEqual(dragon_pair["wiki_page_ids"], ["Gold_Dragon", "Gold_Dragon"])
            self.assertIn("traditional folklore", dragon_pair["passage_text"])
            self.assertEqual(len(synthetic), 4)
            self.assertTrue(all(p["relevance_label"] == "irrelevant" for p in synthetic))
            self.assertTrue(all(p["stance_label"] is None for p in synthetic))
            self.assertTrue(all(p["is_synthetic_label"] is True for p in synthetic))
            self.assertTrue(all(p["split"] == "train" for p in train_pairs))
            self.assertTrue(all(p["split"] == "validation" for p in dev_pairs))
            self.assertEqual(manifest["input_claim_totals"], {"train": 3, "validation": 1})
            self.assertTrue((out / "missing_evidence.jsonl").is_file())
            self.assertTrue((out / "dataset_manifest.json").is_file())
            self.assertEqual(validate_split(out / "train_pairs.jsonl", "train")["pairs_total"], len(train_pairs))
            self.assertEqual(validate_split(out / "validation_pairs.jsonl", "validation")["pairs_total"], len(dev_pairs))
            train_negative_pages = {p["wiki_page_ids"][0] for p in synthetic if p["split"] == "train"}
            validation_negative_pages = {p["wiki_page_ids"][0] for p in synthetic if p["split"] == "validation"}
            self.assertTrue(train_negative_pages.isdisjoint(validation_negative_pages))

    def test_duplicate_ids_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fever = root / "fever"
            fever.mkdir()
            self.write_jsonl(fever / "train.jsonl", [{"id": 1, "claim": "A claim", "label": "NOT ENOUGH INFO", "evidence": []}])
            self.write_jsonl(fever / "shared_task_dev.jsonl", [{"id": 1, "claim": "Another claim", "label": "NOT ENOUGH INFO", "evidence": []}])
            with self.assertRaises(DatasetError):
                prepare_dataset(fever, root / "missing_wiki", root / "out", negatives_per_claim=0)

    def test_missing_gold_sentence_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fever = root / "fever"
            wiki = root / "wiki"
            out = root / "out"
            wiki.mkdir()
            fever.mkdir()
            self.write_jsonl(fever / "train.jsonl", [
                {"id": 1, "claim": "The blue whale is a mammal.", "label": "SUPPORTS", "evidence": [[[1, 2, "Missing_Page", 0]]]}
            ])
            self.write_jsonl(fever / "shared_task_dev.jsonl", [])
            self.write_jsonl(wiki / "wiki-000.jsonl", [
                {"id": "Random", "text": "", "lines": "0\tA completely unrelated sentence about pottery and music."}
            ])
            manifest = prepare_dataset(fever, wiki, out, negatives_per_claim=0, reservoir_size=0)
            self.assertEqual(manifest["claims_with_no_resolved_gold_evidence"]["train"], 1)
            missing = json.loads((out / "missing_evidence.jsonl").read_text().splitlines()[0])
            self.assertEqual(missing["reason"], "page_not_found")
            pairs = (out / "train_pairs.jsonl").read_text()
            self.assertEqual(pairs, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
