
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

LABEL = "category_level_1"


def load_split(name):
    path = DATA_DIR / f"{name}.jsonl"
    df = pd.read_json(path, lines=True)

    if df["text"].isna().any() or df[LABEL].isna().any():
        raise ValueError(f"Missing text or labels in {path}")

    return df


def build_models():
    common = dict(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
    )

    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(**common)),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=42,
            )),
        ]),
    }


def main():
    train = load_split("train")
    validation = load_split("validation")

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results = []

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(train["text"], train[LABEL])
        predictions = model.predict(validation["text"])

        metrics = {
            "model": name,
            "accuracy": accuracy_score(
                validation[LABEL], predictions
            ),
            "macro_f1": f1_score(
                validation[LABEL], predictions,
                average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                validation[LABEL], predictions,
                average="weighted", zero_division=0
            ),
        }
        results.append(metrics)

        print(metrics)
        print(classification_report(
            validation[LABEL],
            predictions,
            zero_division=0,
        ))

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    results_df.to_csv(
        REPORT_DIR / "validation_comparison.csv",
        index=False,
    )

    print("\nValidation comparison:")
    print(results_df.to_string(index=False))
    print("\nModels saved in:", MODEL_DIR)
    print("Comparison saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()

