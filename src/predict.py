
import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "headline_svm.joblib"


def rank_decision_scores(model, text, limit=3):
    """Return top raw LinearSVC margins for multiclass or binary classifiers."""
    scores = model.decision_function([text])
    classes = list(model.classes_)
    if getattr(scores, "ndim", 1) == 1 and len(classes) == 2:
        # Binary LinearSVC returns one signed margin: positive favors classes_[1].
        margin = float(scores[0])
        class_scores = [(classes[0], -margin), (classes[1], margin)]
    else:
        score_vector = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        class_scores = list(zip(classes, score_vector))
    return sorted(
        class_scores,
        key=lambda item: float(item[1]),
        reverse=True,
    )[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Classify a news headline using the MN-DS headline model."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="News headline text to classify (not a full article body)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="UTF-8 text file containing a news headline",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Model joblib path (default: headline_svm.joblib).",
    )
    args = parser.parse_args()

    if bool(args.text) == bool(args.file):
        parser.error("Provide exactly one of TEXT or --file.")

    model_path = args.model or DEFAULT_MODEL_PATH
    
    if not model_path.is_file():
        parser.error(
            f"Model not found: {model_path}. Train it first or pass --model PATH."
        )

    text = args.file.read_text(encoding="utf-8").strip() if args.file else args.text.strip()

    if not text:
        parser.error("Input text cannot be empty.")

    model = joblib.load(model_path)
    print(f"Model: {model_path}")
    prediction = model.predict([text])[0]

    print(f"Predicted category: {prediction}")

    # LinearSVC decision scores are raw margins, not calibrated probabilities.
    ranked = rank_decision_scores(model, text, limit=3)

    print("\nTop 3 model scores (not probabilities):")
    for label, score in ranked:
        print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()

