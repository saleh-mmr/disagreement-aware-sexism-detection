import os
import json
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python3 scripts/analyze_by_language.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEV_PATH, PREDICTION_OUTPUT
from src.data.preprocessing import load_annotated_data
from src.engine.metrics import compute_classification_metrics


OUTPUT_DIR = "outputs/analysis"


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_to_label(prediction_item):
    hard_label = prediction_item["hard_label"]

    if hard_label == "YES":
        return 1
    elif hard_label == "NO":
        return 0
    else:
        raise ValueError(f"Unknown hard label: {hard_label}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prediction_path = os.path.join(
        PREDICTION_OUTPUT,
        "dev_task1_ensemble_predictions.json"
    )

    print("Loading development data...")
    dev_df = load_annotated_data(DEV_PATH)

    print("Loading ensemble predictions...")
    predictions = load_predictions(prediction_path)

    rows = []

    for _, row in dev_df.iterrows():
        tweet_id = str(row["id_EXIST"])

        if tweet_id not in predictions:
            continue

        gold_label = int(row["hard_label"])
        predicted_label = prediction_to_label(predictions[tweet_id])

        no_prob = predictions[tweet_id]["soft_label"]["NO"]
        yes_prob = predictions[tweet_id]["soft_label"]["YES"]

        rows.append({
            "id_EXIST": tweet_id,
            "lang": row["lang"],
            "text": row["text"],
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "gold_label_name": "YES" if gold_label == 1 else "NO",
            "predicted_label_name": "YES" if predicted_label == 1 else "NO",
            "NO_probability": no_prob,
            "YES_probability": yes_prob,
            "confidence": max(no_prob, yes_prob),
            "correct": int(gold_label == predicted_label)
        })

    analysis_df = pd.DataFrame(rows)

    full_output_path = os.path.join(
        OUTPUT_DIR,
        "language_level_predictions.csv"
    )
    analysis_df.to_csv(full_output_path, index=False)

    summary_rows = []

    for lang, group_df in analysis_df.groupby("lang"):
        true_labels = group_df["gold_label"].values
        pred_labels = group_df["predicted_label"].values

        metrics = compute_classification_metrics(true_labels, pred_labels)

        summary_rows.append({
            "language": lang,
            "num_examples": len(group_df),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "average_confidence": group_df["confidence"].mean(),
            "num_errors": int((group_df["correct"] == 0).sum())
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("language")

    summary_output_path = os.path.join(
        OUTPUT_DIR,
        "language_level_summary.csv"
    )
    summary_df.to_csv(summary_output_path, index=False)

    print("\n===== Language-Based Evaluation =====")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(full_output_path)
    print(summary_output_path)


if __name__ == "__main__":
    main()