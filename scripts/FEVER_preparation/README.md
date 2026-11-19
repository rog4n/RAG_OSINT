# FEVER v1.0 pair preparation for OSINT_final

This bundle converts the FEVER v1.0 labelled training and development files plus the extracted Wikipedia corpus into claim–sentence pairs. It is deliberately separate from the existing OSINT pipeline.

## What the scripts create

- `train_pairs.jsonl`
- `validation_pairs.jsonl`
- `missing_evidence.jsonl`
- `dataset_manifest.json`

Each pair includes the requested fields `claim_text`, `passage_text`, `relevance_label`, and `stance_label`, plus claim ID, original FEVER claim label, split, one or more Wikipedia page/sentence locators, evidence-group IDs, and label provenance.

### Label policy (read this before training)

1. For FEVER `SUPPORTS` and `REFUTES` claims, each complete FEVER evidence group becomes one **gold passage pair**. All sentences in a group are concatenated because a multi-sentence group may be jointly sufficient:
   - relevance: `relevant`
   - stance: `supports` or `contradicts`
   - `label_origin`: `fever_gold_evidence`
2. Each claim can also receive randomly sampled Wikipedia sentences from pages that are not used as gold-evidence pages anywhere in FEVER train/dev. A negative is accepted only if it has **zero shared non-stopword tokens** with the claim:
   - relevance: `irrelevant`
   - stance: `insufficient`
   - `label_origin`: `synthetic_random_negative`
   - `is_synthetic_label`: `true`
3. FEVER `NOT ENOUGH INFO` is a **claim-level** label and has no gold evidence sentence. The script does not pretend that FEVER supplies passage-level `insufficient` labels. NEI claims receive only the explicitly synthetic negatives described above.
4. Missing Wikipedia pages/sentences are written to `missing_evidence.jsonl` and are not assigned guessed labels.
5. FEVER's unlabelled test file is not used. The source files do not provide publisher or publication-date metadata, so the script does not invent it.

**Important modelling limitation:** the synthetic negatives are easy, weakly labelled examples. They are useful for an initial relevance baseline, but they do not replace human-annotated hard negatives or on-topic passages labelled insufficient. In particular, the resulting data does not independently teach the full distinction between an irrelevant passage and a relevant-but-inconclusive passage. Keep gold and synthetic metrics separate when evaluating.

## Install location

Copy these three Python files into a new folder in your project, for example:

```text
OSINT_final/
  scripts/fever_preparation/prepare_fever_pairs.py
  scripts/fever_preparation/validate_fever_pairs.py
  scripts/fever_preparation/test_prepare_fever_pairs.py
```

No third-party Python packages are required. The scripts use only the Python standard library. Run them from the project root, using the same Python interpreter you use for the main OSINT project.

## Expected input layout

This matches the files you downloaded and extracted:

```text
data/fever/
  train.jsonl
  shared_task_dev.jsonl
  wiki-pages/
    wiki-pages/
      wiki-000.jsonl
      wiki-001.jsonl
      ...
```

The `wiki-pages.zip` must already be extracted. The script recursively discovers `wiki-*.jsonl` shards and ignores macOS `__MACOSX` metadata folders.

## 1. Run the tests first

From `OSINT_final/`:

```bash
python -m unittest discover -s scripts/fever_preparation -p 'test_*.py' -v
```

The tests use tiny temporary fixtures. They do not read, modify, or require your 7.1 GB extracted Wikipedia corpus.

## 2. Build the pair files

From `OSINT_final/`:

```bash
python scripts/fever_preparation/prepare_fever_pairs.py \
  --fever-dir data/fever \
  --wiki-dir data/fever/wiki-pages/wiki-pages \
  --output-dir data/fever/processed_pairs \
  --negatives-per-claim 1 \
  --reservoir-size 50000 \
  --seed 42
```

This reads the train/dev claim files and scans the Wikipedia shards once. The scan can take time because the extracted corpus is large. The script prints a summary when finished. It writes only under `data/fever/processed_pairs/`; it does not update `data/new_articles.db`, any existing model, or any existing pipeline script.

To build gold evidence pairs only (no generated negatives), use `--negatives-per-claim 0`. This is useful for checking evidence reconstruction first, but does not create an `irrelevant` training class.

## 3. Validate the generated files

```bash
python scripts/fever_preparation/validate_fever_pairs.py \
  --input-dir data/fever/processed_pairs
```

Expected final line: `VALIDATION PASSED`.

## Output label values

| Field | Values |
|---|---|
| `relevance_label` | `relevant`, `irrelevant` |
| `stance_label` | `supports`, `contradicts`, `insufficient` |
| `claim_level_label` | `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` |
| `label_origin` | `fever_gold_evidence`, `synthetic_random_negative` |

`evidence_group_ids` are zero-based indices into the original FEVER `evidence` list. Multiple alternative groups are retained as separate pairs. Each pair stores `wiki_page_ids` and `wiki_sentence_ids` arrays, which can contain more than one locator. If any sentence in an evidence group cannot be resolved, the entire group is excluded rather than emitting a partial, potentially insufficient passage.

## Resource and safety notes

- The Wikipedia archive expands to several GB. The script streams the shards; it does not load the full corpus into RAM.
- It retains only the FEVER-referenced gold sentences plus a bounded random sentence reservoir (default 50,000 passages).
- The script fails on malformed JSONL rather than silently ignoring a corrupted input row.
- Re-running overwrites only the four named outputs inside `processed_pairs`; it does not delete the input dataset or touch other project files.
- Do not train a final model from these pairs before inspecting `dataset_manifest.json` and running the validator. Treat synthetic negative results as weak-label baseline results, not FEVER benchmark scores.
