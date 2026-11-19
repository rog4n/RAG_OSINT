"""Train the final headline-only LinearSVC using validation-selected parameters."""
from __future__ import annotations

import ast
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"


def parse_bool(value) -> bool:
    """Parse booleans from pandas/CSV values without bool('False') pitfalls."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Unrecognized boolean value for char_features: {value!r}")


def main() -> None:
    results_path = REPORT_DIR / "headline_tuning_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing tuning results: {results_path}")
    results = pd.read_csv(results_path)
    required_results = {"macro_f1", "accuracy", "C", "word_ngrams", "min_df", "char_features"}
    missing = required_results - set(results.columns)
    if missing:
        raise ValueError(f"Tuning results missing columns: {', '.join(sorted(missing))}")
    if results.empty:
        raise ValueError("Tuning results contain no candidate configurations.")

    best = results.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "validation.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Expected data/processed/train.jsonl and validation.jsonl.")
    train = pd.read_json(train_path, lines=True)
    val = pd.read_json(val_path, lines=True)
    required_data = {"title", "category_level_1"}
    for label, frame in (("train", train), ("validation", val)):
        absent = required_data - set(frame.columns)
        if absent:
            raise ValueError(f"{label} split missing columns: {', '.join(sorted(absent))}")
        frame["title"] = frame["title"].fillna("").astype(str).str.strip()
        frame["category_level_1"] = frame["category_level_1"].fillna("").astype(str).str.strip()
        frame.drop(frame[(frame.title == "") | (frame.category_level_1 == "")].index, inplace=True)
        if frame.empty:
            raise ValueError(f"{label} split has no usable rows.")

    combined = pd.concat([train, val], ignore_index=True)
    X = combined["title"]
    y = combined["category_level_1"]
    C = float(best["C"])
    word_ngrams = ast.literal_eval(str(best["word_ngrams"]))
    if not isinstance(word_ngrams, tuple) or len(word_ngrams) != 2:
        raise ValueError(f"Invalid word_ngrams configuration: {best['word_ngrams']!r}")
    min_df = int(best["min_df"])
    if min_df < 1 or C <= 0:
        raise ValueError("C and min_df must be positive.")
    use_char = parse_bool(best["char_features"])

    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=word_ngrams, min_df=min_df,
        max_features=250_000, sublinear_tf=True, strip_accents="unicode",
    )
    if use_char:
        features = FeatureUnion([
            ("word", word_vectorizer),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                max_features=150_000, sublinear_tf=True, strip_accents="unicode",
            )),
        ])
    else:
        features = word_vectorizer

    model = Pipeline([
        ("tfidf", features),
        ("classifier", LinearSVC(C=C, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "headline_svm.joblib"
    joblib.dump(model, output_path)
    print(f"Best validation macro F1: {float(best['macro_f1']):.4f}")
    print(f"Best validation accuracy: {float(best['accuracy']):.4f}")
    print(f"Character features enabled: {use_char}")
    print(f"Training records (train + validation): {len(combined)}")
    print(f"Saved model: {output_path}")


if __name__ == "__main__":
    main()
