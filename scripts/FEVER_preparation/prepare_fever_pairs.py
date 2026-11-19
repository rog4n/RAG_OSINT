#!/usr/bin/env python3
"""Prepare FEVER v1.0 claim/passage pairs for the OSINT project.

This script uses only the Python standard library. It reads the official FEVER
train and labelled development JSONL files plus the extracted Wikipedia pages.
It writes pair-level JSONL files; it never touches the project's SQLite database
or existing model files.

Important label semantics:
- Gold FEVER evidence sentences become relevant + supports/contradicts pairs.
- Random Wikipedia sentences with zero shared content words become explicitly
  synthetic irrelevant pairs. Their stance label is null because an unrelated
  passage does not provide a valid passage-level stance annotation.
- FEVER NOT ENOUGH INFO is a claim-level label without annotated evidence. It
  does not magically label an arbitrary passage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
STANCE_MAP = {
    "SUPPORTS": "supports",
    "REFUTES": "contradicts",
}

# A conservative English stop-word list for screening easy random negatives.
# A candidate is accepted only when no non-stopword token is shared with claim.
STOPWORDS = frozenset("""
a an the and or but if then else when while of at by for from in into on onto to
with without as is am are was were be been being do does did have has had can
could should would may might must will shall this that these those it its they
them their he she his her we our you your i me my mine yours not no nor so than
too very also just about over under after before during between through above
below up down out off again further once here there where why how what which who
whom whose all any both each few more most other some such only own same than
s t re ve d ll m
""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+")


class DatasetError(RuntimeError):
    """Raised when an input dataset is malformed or incomplete."""


@dataclass(frozen=True)
class EvidenceRef:
    page_id: str
    sentence_id: int
    group_index: int


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    claim_text: str
    label: str
    split: str
    evidence_refs: Tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class Passage:
    page_ids: Tuple[str, ...]
    sentence_ids: Tuple[int, ...]
    text: str


def content_tokens(text: str) -> Set[str]:
    """Return lowercase content tokens used only for negative screening."""
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1}


def parse_wiki_lines(lines: Any, wanted_ids: Optional[Set[int]] = None) -> Iterator[Tuple[int, str]]:
    """Parse the FEVER wiki-pages `lines` field into (sentence_id, sentence).

    FEVER lines are tab-delimited: sentence_id, sentence text, then optional
    entity annotations. Only the first two columns are used here.
    """
    if not isinstance(lines, str):
        return
    for raw in lines.splitlines():
        if not raw:
            continue
        parts = raw.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            sentence_id = int(parts[0])
        except (TypeError, ValueError):
            continue
        if wanted_ids is not None and sentence_id not in wanted_ids:
            continue
        sentence = parts[1].strip()
        if sentence:
            yield sentence_id, sentence


def parse_evidence_groups(raw_evidence: Any, row_context: str) -> Tuple[EvidenceRef, ...]:
    """Flatten valid page/sentence references while retaining alternative group IDs."""
    if raw_evidence is None:
        return ()
    if not isinstance(raw_evidence, list):
        raise DatasetError(f"{row_context}: 'evidence' must be a list")
    refs: List[EvidenceRef] = []
    for group_index, group in enumerate(raw_evidence):
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page_id = item[2]
            sentence_id = item[3]
            # FEVER NEI examples commonly use [annotator, null, null, null].
            if page_id is None or sentence_id is None:
                continue
            try:
                sentence_id_int = int(sentence_id)
            except (TypeError, ValueError):
                continue
            refs.append(EvidenceRef(str(page_id), sentence_id_int, group_index))
    return tuple(refs)


def read_claim_file(path: Path, split: str) -> List[ClaimRecord]:
    if not path.is_file():
        raise DatasetError(f"Missing required input file: {path}")
    claims: List[ClaimRecord] = []
    seen_ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            context = f"{path.name}:{line_number}"
            for field in ("id", "claim", "label"):
                if field not in row:
                    raise DatasetError(f"{context}: missing required field {field!r}")
            label = str(row["label"]).strip().upper()
            if label not in FEVER_LABELS:
                raise DatasetError(f"{context}: unsupported FEVER label {label!r}")
            claim_id = str(row["id"])
            if claim_id in seen_ids:
                raise DatasetError(f"{context}: duplicate claim id {claim_id!r} within {split}")
            seen_ids.add(claim_id)
            claim_text = str(row["claim"]).strip()
            if not claim_text:
                raise DatasetError(f"{context}: empty claim text")
            evidence_refs = parse_evidence_groups(row.get("evidence", []), context)
            claims.append(ClaimRecord(claim_id, claim_text, label, split, evidence_refs))
    return claims


def discover_wiki_shards(wiki_dir: Path) -> List[Path]:
    if not wiki_dir.is_dir():
        raise DatasetError(f"Wikipedia directory does not exist: {wiki_dir}")
    shards = sorted(
        p for p in wiki_dir.rglob("wiki-*.jsonl")
        if "__MACOSX" not in p.parts and p.is_file()
    )
    if not shards:
        raise DatasetError(
            f"No wiki-*.jsonl shards found under {wiki_dir}. "
            "Expected the extracted wiki-pages/wiki-pages directory."
        )
    return shards


def stable_id(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def negative_pool_split(page_id: str) -> str:
    """Assign each non-evidence Wikipedia page to exactly one pair split."""
    bucket = int(stable_id("negative-page-split", page_id)[:8], 16) % 2
    return "train" if bucket == 0 else "validation"


def scan_wikipedia(
    shards: Sequence[Path],
    needed_sentences: Dict[str, Set[int]],
    reservoir_size: int,
    seed: int,
) -> Tuple[Dict[Tuple[str, int], str], Dict[str, List[Passage]], Dict[str, int], Set[str]]:
    """Stream shards once; build page-disjoint train and validation negative reservoirs.

    ``reservoir_size`` is the maximum size PER split. A page is deterministically
    assigned to only one split, preventing synthetic passage reuse across train
    and validation.
    """
    evidence_text: Dict[Tuple[str, int], str] = {}
    found_pages: Set[str] = set()
    pools: Dict[str, List[Passage]] = {"train": [], "validation": []}
    pool_rngs = {
        "train": random.Random(seed),
        "validation": random.Random(seed ^ 0x5EED5EED),
    }
    eligible_seen = {"train": 0, "validation": 0}
    pages_seen = 0
    sentences_seen = 0
    malformed_wiki_rows = 0

    for shard_index, shard in enumerate(shards, start=1):
        if shard_index == 1 or shard_index % 10 == 0 or shard_index == len(shards):
            print(f"Scanning Wikipedia shard {shard_index}/{len(shards)}: {shard.name}", flush=True)
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"Invalid JSON in {shard}:{line_number}: {exc}") from exc
                page_id_value = record.get("id")
                if page_id_value is None:
                    malformed_wiki_rows += 1
                    continue
                page_id = str(page_id_value)
                pages_seen += 1

                if page_id in needed_sentences:
                    found_pages.add(page_id)
                    for sentence_id, sentence in parse_wiki_lines(record.get("lines", ""), needed_sentences[page_id]):
                        evidence_text[(page_id, sentence_id)] = sentence
                    # Do not sample any sentence from pages used as gold evidence.
                    continue

                if reservoir_size <= 0:
                    continue
                split_for_page = negative_pool_split(page_id)
                pool = pools[split_for_page]
                pool_rng = pool_rngs[split_for_page]
                for sentence_id, sentence in parse_wiki_lines(record.get("lines", "")):
                    sentences_seen += 1
                    if len(sentence) < 25:
                        continue
                    eligible_seen[split_for_page] += 1
                    candidate = Passage((page_id,), (sentence_id,), sentence)
                    if len(pool) < reservoir_size:
                        pool.append(candidate)
                    else:
                        replacement_index = pool_rng.randrange(eligible_seen[split_for_page])
                        if replacement_index < reservoir_size:
                            pool[replacement_index] = candidate

    metrics = {
        "wiki_shards_scanned": len(shards),
        "wiki_pages_seen": pages_seen,
        "wiki_sentences_seen_in_non_evidence_pages": sentences_seen,
        "negative_pool_eligible_sentences_seen": sum(eligible_seen.values()),
        "negative_pool_eligible_sentences_seen_by_split": eligible_seen,
        "negative_pool_size_by_split": {split: len(pool) for split, pool in pools.items()},
        "negative_pool_size_total": sum(len(pool) for pool in pools.values()),
        "malformed_wiki_rows_missing_id": malformed_wiki_rows,
        "gold_evidence_pages_found": len(found_pages),
    }
    return evidence_text, pools, metrics, found_pages


def choose_negative(
    claim_text: str,
    pool: Sequence[Passage],
    rng: random.Random,
    already_used: Set[Tuple[str, int]],
    max_attempts: int,
) -> Optional[Passage]:
    """Choose an easy random negative with zero shared content-token overlap."""
    if not pool:
        return None
    claim_terms = content_tokens(claim_text)
    if not claim_terms:
        return None
    for _ in range(max_attempts):
        passage = pool[rng.randrange(len(pool))]
        key = (passage.page_ids[0], passage.sentence_ids[0])
        if key in already_used:
            continue
        if claim_terms.isdisjoint(content_tokens(passage.text)):
            already_used.add(key)
            return passage
    return None


def make_pair(
    claim: ClaimRecord,
    passage: Passage,
    relevance: str,
    stance: Optional[str],
    label_origin: str,
    evidence_group_ids: Optional[Sequence[int]] = None,
    negative_index: Optional[int] = None,
) -> Dict[str, Any]:
    identity_suffix = (
        tuple(sorted(set(evidence_group_ids or [])))
        if label_origin == "fever_gold_evidence"
        else negative_index
    )
    pair_id = stable_id(
        claim.split, claim.claim_id, passage.page_ids, passage.sentence_ids,
        label_origin, identity_suffix,
    )
    return {
        "pair_id": pair_id,
        "claim_id": claim.claim_id,
        "claim_text": claim.claim_text,
        "passage_text": passage.text,
        "relevance_label": relevance,
        "stance_label": stance,
        "claim_level_label": claim.label,
        "split": claim.split,
        "passage_source": "FEVER_WIKIPEDIA",
        "wiki_page_ids": list(passage.page_ids),
        "wiki_sentence_ids": list(passage.sentence_ids),
        "evidence_group_ids": sorted(set(evidence_group_ids or [])),
        "label_origin": label_origin,
        "is_synthetic_label": label_origin != "fever_gold_evidence",
    }


def write_jsonl_row(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_dataset(
    fever_dir: Path,
    wiki_dir: Path,
    output_dir: Path,
    negatives_per_claim: int = 1,
    reservoir_size: int = 50000,
    seed: int = 42,
    max_negative_attempts: int = 100,
) -> Dict[str, Any]:
    if negatives_per_claim < 0:
        raise DatasetError("negatives_per_claim must be >= 0")
    if reservoir_size < 0:
        raise DatasetError("reservoir_size must be >= 0")
    if max_negative_attempts < 1:
        raise DatasetError("max_negative_attempts must be >= 1")

    train_claims = read_claim_file(fever_dir / "train.jsonl", "train")
    validation_claims = read_claim_file(fever_dir / "shared_task_dev.jsonl", "validation")
    all_claims = train_claims + validation_claims
    all_ids: Set[str] = set()
    for claim in all_claims:
        if claim.claim_id in all_ids:
            raise DatasetError(
                f"Claim id {claim.claim_id!r} appears in both train and validation; "
                "refusing a potentially leaked split."
            )
        all_ids.add(claim.claim_id)

    needed_sentences: Dict[str, Set[int]] = defaultdict(set)
    for claim in all_claims:
        for ref in claim.evidence_refs:
            needed_sentences[ref.page_id].add(ref.sentence_id)

    shards = discover_wiki_shards(wiki_dir)
    evidence_text, negative_pools, scan_metrics, found_wiki_pages = scan_wikipedia(
        shards, needed_sentences, reservoir_size, seed
    )
    if negatives_per_claim > 0 and not any(negative_pools.values()):
        raise DatasetError(
            "No candidate negative passages were collected. Check that the Wikipedia "
            "shards are extracted correctly and contain populated 'lines' fields."
        )
    if negatives_per_claim > 0:
        empty_splits = [split for split, pool in negative_pools.items() if not pool]
        if empty_splits:
            raise DatasetError(
                "No synthetic-negative reservoir was built for split(s): " + ", ".join(empty_splits) +
                ". Increase the Wikipedia corpus/reservoir or set --negatives-per-claim 0."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_tmp = output_dir / "train_pairs.jsonl.tmp"
    validation_tmp = output_dir / "validation_pairs.jsonl.tmp"
    missing_tmp = output_dir / "missing_evidence.jsonl.tmp"
    manifest_tmp = output_dir / "dataset_manifest.json.tmp"

    negative_rngs = {
        "train": random.Random(seed ^ 0x13579BDF),
        "validation": random.Random(seed ^ 0x2468ACE0),
    }
    counts: Dict[str, Counter] = {
        "train": Counter(),
        "validation": Counter(),
    }
    input_label_counts: Dict[str, Counter] = {
        "train": Counter(c.label for c in train_claims),
        "validation": Counter(c.label for c in validation_claims),
    }
    missing_counts = Counter()
    no_resolved_gold: Counter = Counter()
    claim_counts = {"train": len(train_claims), "validation": len(validation_claims)}

    try:
        with train_tmp.open("w", encoding="utf-8") as train_out, \
             validation_tmp.open("w", encoding="utf-8") as validation_out, \
             missing_tmp.open("w", encoding="utf-8") as missing_out:
            outputs = {"train": train_out, "validation": validation_out}
            for claim in all_claims:
                out = outputs[claim.split]
                # Each FEVER evidence group is an alternative sufficient evidence set.
                # Keep all sentences in a group together: an individual sentence may
                # not independently entail or refute a multi-hop claim.
                refs_by_group: Dict[int, List[EvidenceRef]] = defaultdict(list)
                for ref in claim.evidence_refs:
                    refs_by_group[ref.group_index].append(ref)

                resolved_groups_for_claim = 0
                if claim.label in STANCE_MAP:
                    for group_index, group_refs in sorted(refs_by_group.items()):
                        unique_refs: List[EvidenceRef] = []
                        seen_refs: Set[Tuple[str, int]] = set()
                        for ref in group_refs:
                            key = (ref.page_id, ref.sentence_id)
                            if key not in seen_refs:
                                seen_refs.add(key)
                                unique_refs.append(ref)

                        group_sentences: List[str] = []
                        group_missing = []
                        for ref in unique_refs:
                            sentence = evidence_text.get((ref.page_id, ref.sentence_id))
                            if sentence is None:
                                reason = "page_found_sentence_not_found" if ref.page_id in found_wiki_pages else "page_not_found"
                                missing_counts[reason] += 1
                                group_missing.append(ref)
                                write_jsonl_row(missing_out, {
                                    "split": claim.split,
                                    "claim_id": claim.claim_id,
                                    "claim_text": claim.claim_text,
                                    "fever_label": claim.label,
                                    "evidence_group_id": group_index,
                                    "wiki_page_id": ref.page_id,
                                    "wiki_sentence_id": ref.sentence_id,
                                    "reason": reason,
                                })
                            else:
                                group_sentences.append(sentence)

                        # Never emit a partial evidence group as if it were sufficient.
                        if group_missing or not group_sentences:
                            counts[claim.split]["incomplete_evidence_groups"] += 1
                            continue

                        group_passage = Passage(
                            tuple(ref.page_id for ref in unique_refs),
                            tuple(ref.sentence_id for ref in unique_refs),
                            " ".join(group_sentences),
                        )
                        pair = make_pair(
                            claim,
                            group_passage,
                            "relevant",
                            STANCE_MAP[claim.label],
                            "fever_gold_evidence",
                            [group_index],
                        )
                        write_jsonl_row(out, pair)
                        counts[claim.split]["pairs_total"] += 1
                        counts[claim.split]["relevance_relevant"] += 1
                        counts[claim.split][f"stance_{STANCE_MAP[claim.label]}"] += 1
                        counts[claim.split]["gold_evidence_pairs"] += 1
                        resolved_groups_for_claim += 1
                    if not resolved_groups_for_claim:
                        no_resolved_gold[claim.split] += 1
                elif claim.evidence_refs:
                    # Do not silently convert unexpected evidence on an NEI record into gold labels.
                    counts[claim.split]["nei_records_with_nonnull_evidence_refs"] += 1

                already_used: Set[Tuple[str, int]] = set()
                for negative_index in range(negatives_per_claim):
                    negative = choose_negative(
                        claim.claim_text, negative_pools[claim.split], negative_rngs[claim.split],
                        already_used, max_negative_attempts
                    )
                    if negative is None:
                        counts[claim.split]["negative_sampling_shortfall"] += 1
                        continue
                    pair = make_pair(
                        claim,
                        negative,
                        "irrelevant",
                        None,
                        "synthetic_random_negative",
                        negative_index=negative_index,
                    )
                    write_jsonl_row(out, pair)
                    counts[claim.split]["pairs_total"] += 1
                    counts[claim.split]["relevance_irrelevant"] += 1
                    counts[claim.split]["stance_unlabelled_synthetic_pairs"] += 1
                    counts[claim.split]["synthetic_negative_pairs"] += 1

        # Publish only after all outputs have been written successfully.
        os.replace(train_tmp, output_dir / "train_pairs.jsonl")
        os.replace(validation_tmp, output_dir / "validation_pairs.jsonl")
        os.replace(missing_tmp, output_dir / "missing_evidence.jsonl")

        manifest = {
            "dataset": "FEVER v1.0",
            "preparation_version": 2,
            "description": "Claim/passage pairs reconstructed from FEVER gold evidence plus explicitly synthetic random negatives.",
            "input_files": {
                "train": str(fever_dir / "train.jsonl"),
                "validation": str(fever_dir / "shared_task_dev.jsonl"),
                "wikipedia_directory": str(wiki_dir),
            },
            "output_files": [
                "train_pairs.jsonl",
                "validation_pairs.jsonl",
                "missing_evidence.jsonl",
            ],
            "schema": {
                "claim_text": "Original FEVER claim",
                "passage_text": "One complete FEVER evidence group concatenated into a passage, or one synthetic negative sentence",
                "relevance_label": ["relevant", "irrelevant"],
                "stance_label": "supports/contradicts for gold evidence; null for synthetic irrelevant negatives",
                "claim_level_label": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
                "label_origin": ["fever_gold_evidence", "synthetic_random_negative"],
                "wiki_page_ids": "Wikipedia page IDs corresponding positionally to wiki_sentence_ids",
                "wiki_sentence_ids": "Sentence IDs corresponding positionally to wiki_page_ids",
                "is_synthetic_label": "True only for generated random negatives",
            },
            "labeling_policy": {
                "gold_positive_pairs": "Each complete FEVER-annotated evidence group for SUPPORTS/REFUTES is emitted as one concatenated passage; groups are alternative sufficient evidence sets and are preserved as zero-based group indices. Partial groups are excluded.",
                "synthetic_negative_pairs": "Random Wikipedia sentences from pages not used as gold evidence anywhere in train/dev, accepted only when they share zero non-stopword tokens with the claim. Labelled irrelevant for relevance and null/unlabelled for stance; these are weak negatives, not human annotations.",
                "negative_split_isolation": "Non-evidence Wikipedia pages are deterministically assigned to exactly one of train or validation; synthetic passage locators cannot cross the split boundary.",
                "not_enough_info": "FEVER NOT ENOUGH INFO is claim-level only. No passage-level insufficient examples are inferred from it.",
                "unresolved_gold_evidence": "Excluded from pair files and recorded in missing_evidence.jsonl; never assigned a guessed label.",
            },
            "configuration": {
                "negatives_per_claim": negatives_per_claim,
                "negative_reservoir_size_per_split": reservoir_size,
                "random_seed": seed,
                "max_negative_attempts_per_negative": max_negative_attempts,
                "negative_overlap_policy": "zero shared non-stopword tokens",
                "negative_pool_split_policy": "stable SHA-256 page-ID assignment; train and validation pages are disjoint",
            },
            "input_claim_counts": {
                split: {label: int(input_label_counts[split][label]) for label in sorted(FEVER_LABELS)}
                for split in ("train", "validation")
            },
            "input_claim_totals": claim_counts,
            "pair_counts": {
                split: {key: int(value) for key, value in sorted(counts[split].items())}
                for split in ("train", "validation")
            },
            "claims_with_no_resolved_gold_evidence": {
                split: int(no_resolved_gold[split]) for split in ("train", "validation")
            },
            "missing_evidence_counts": {key: int(value) for key, value in sorted(missing_counts.items())},
            "wikipedia_scan": scan_metrics,
            "notes": [
                "FEVER source files provide no publisher or publication-date fields; none are fabricated.",
                "The supplied FEVER test split is unlabelled and is intentionally not used here.",
                "Synthetic negatives are suitable only for an initial weak-label relevance baseline; they do not replace annotated hard negatives or passage-level insufficient examples.",
                "Synthetic negative pages are disjoint between train and validation. Report gold-evidence and synthetic-negative metrics separately.",
                "Synthetic negatives have null stance labels and must be excluded from stance-model training and evaluation.",
            ],
        }
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(manifest_tmp, output_dir / "dataset_manifest.json")
        return manifest
    finally:
        for temporary in (train_tmp, validation_tmp, missing_tmp, manifest_tmp):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build FEVER claim/passage pair JSONL files for relevance and stance experiments."
    )
    parser.add_argument("--fever-dir", type=Path, default=Path("data/fever"),
                        help="Directory containing train.jsonl and shared_task_dev.jsonl (default: data/fever).")
    parser.add_argument("--wiki-dir", type=Path, default=None,
                        help="Extracted Wikipedia shard directory. Defaults to FEVER_DIR/wiki-pages/wiki-pages.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to FEVER_DIR/processed_pairs.")
    parser.add_argument("--negatives-per-claim", type=int, default=1,
                        help="Number of synthetic random negatives per claim (default: 1; use 0 to disable).")
    parser.add_argument("--reservoir-size", type=int, default=50000,
                        help="Maximum random sentence reservoir size while scanning Wikipedia (default: 50000).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42).")
    parser.add_argument("--max-negative-attempts", type=int, default=100,
                        help="Candidate attempts for each synthetic negative (default: 100).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    fever_dir = args.fever_dir.expanduser().resolve()
    wiki_dir = (args.wiki_dir or (fever_dir / "wiki-pages" / "wiki-pages")).expanduser().resolve()
    output_dir = (args.output_dir or (fever_dir / "processed_pairs")).expanduser().resolve()
    try:
        manifest = prepare_dataset(
            fever_dir=fever_dir,
            wiki_dir=wiki_dir,
            output_dir=output_dir,
            negatives_per_claim=args.negatives_per_claim,
            reservoir_size=args.reservoir_size,
            seed=args.seed,
            max_negative_attempts=args.max_negative_attempts,
        )
    except (DatasetError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("FEVER pair preparation completed.")
    print(f"Output: {output_dir}")
    for split in ("train", "validation"):
        pair_count = manifest["pair_counts"][split].get("pairs_total", 0)
        print(f"{split}: {manifest['input_claim_totals'][split]:,} claims -> {pair_count:,} pairs")
        print(f"  gold evidence pairs: {manifest['pair_counts'][split].get('gold_evidence_pairs', 0):,}")
        print(f"  synthetic negatives: {manifest['pair_counts'][split].get('synthetic_negative_pairs', 0):,}")
        print(f"  negative sampling shortfall: {manifest['pair_counts'][split].get('negative_sampling_shortfall', 0):,}")
    print(f"Missing gold evidence references: {sum(manifest['missing_evidence_counts'].values()):,}")
    print("Important: synthetic negatives are weak relevance labels only; stance_label is null.")
    print("Synthetic negative Wikipedia pages are isolated between train and validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
