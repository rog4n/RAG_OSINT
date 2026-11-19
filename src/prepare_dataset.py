
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "train.jsonl"
OUT_DIR = ROOT / "data" / "processed"
LABEL = "category_level_1"
SEED = 42


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def main():
    records = [
        json.loads(line)
        for line in RAW_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(records)
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df[LABEL] = df[LABEL].fillna("").astype(str).str.strip()
    df = df[(df["title"] != "") & (df[LABEL] != "")].copy()
    df = df.reset_index(drop=True)

    # Union-find: connect records sharing exact title or content.
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen = {}
    for i, row in df.iterrows():
        for field in ("title", "content"):
            key = normalize(row[field])
            if not key:
                continue
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    df["group"] = [find(i) for i in range(len(df))]
    groups = df.groupby("group")[LABEL].agg(
        size="size", label=lambda s: s.mode().iloc[0]
    ).reset_index()

    # Split groups, never individual records. Group-level stratification
    # approximates label proportions; it is not exact for all classes.
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=SEED,
        stratify=groups["label"],
    )
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_groups["label"],
    )

    assignments = {}
    for name, part in (
        ("train", train_groups),
        ("validation", val_groups),
        ("test", test_groups),
    ):
        for group in part["group"]:
            assignments[group] = name

    df["split"] = df["group"].map(assignments)
    df["text"] = df["title"] + "\n\n" + df["content"]
    df["record_id"] = df.apply(
        lambda r: hashlib.sha256(
            (r["title"] + "\0" + r[LABEL]).encode("utf-8")
        ).hexdigest()[:16],
        axis=1,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        c for c in (
            "record_id", "title", "content", "text", LABEL,
            "category_level_2", "source", "date", "url",
        ) if c in df.columns
    ]

    for name in ("train", "validation", "test"):
        part = df[df["split"] == name].copy()
        part[columns].to_json(
            OUT_DIR / f"{name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        print(f"{name}: {len(part)} records, "
              f"{part[LABEL].nunique()} classes")

    print(f"\nInput records: {len(df)}")
    print(f"Duplicate-connected groups: "
          f"{df['group'].nunique()} unique groups")
    print("Splits rebuilt with exact title/content overlap groups kept together.")
    print("Inspect class counts before training.")


if __name__ == "__main__":
    main()
