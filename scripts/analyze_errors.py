import os
import json
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python3 scripts/analyze_errors.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEV_PATH, PREDICTION_OUTPUT
from src.data.preprocessing import load_annotated_data


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


def get_vote_counts(soft_label):
    """
    Converts soft label probabilities back to approximate annotator vote counts.
    EXIST examples have 6 annotators.
    """
    yes_votes = int(round(soft_label["YES"] * 6))
    no_votes = int(round(soft_label["NO"] * 6))
    return yes_votes, no_votes


def get_disagreement_group(yes_votes, no_votes):
    if yes_votes == 0 or no_votes == 0:
        return "full_agreement"
    elif yes_votes == 1 or no_votes == 1:
        return "low_disagreement"
    elif yes_votes == 2 or no_votes == 2:
        return "medium_disagreement"
    else:
        return "high_disagreement"


def classify_error_type(gold_label, predicted_label):
    if gold_label == 0 and predicted_label == 1:
        return "false_positive"
    elif gold_label == 1 and predicted_label == 0:
        return "false_negative"
    else:
        return "correct"


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

        no_prob = float(predictions[tweet_id]["soft_label"]["NO"])
        yes_prob = float(predictions[tweet_id]["soft_label"]["YES"])

        yes_votes, no_votes = get_vote_counts(row["soft_label"])
        disagreement_group = get_disagreement_group(yes_votes, no_votes)

        confidence = max(no_prob, yes_prob)
        margin = abs(yes_prob - no_prob)

        error_type = classify_error_type(gold_label, predicted_label)

        rows.append({
            "id_EXIST": tweet_id,
            "lang": row["lang"],
            "text": row["text"],
            "gold_label": gold_label,
            "gold_label_name": "YES" if gold_label == 1 else "NO",
            "predicted_label": predicted_label,
            "predicted_label_name": "YES" if predicted_label == 1 else "NO",
            "error_type": error_type,
            "correct": int(error_type == "correct"),
            "gold_soft_NO": float(row["soft_label"]["NO"]),
            "gold_soft_YES": float(row["soft_label"]["YES"]),
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "vote_pattern": f"{yes_votes}_YES_{no_votes}_NO",
            "disagreement_group": disagreement_group,
            "predicted_NO_probability": no_prob,
            "predicted_YES_probability": yes_prob,
            "confidence": confidence,
            "margin": margin
        })

    analysis_df = pd.DataFrame(rows)

    all_predictions_path = os.path.join(
        OUTPUT_DIR,
        "dev_predictions_with_error_labels.csv"
    )
    analysis_df.to_csv(all_predictions_path, index=False)

    errors_df = analysis_df[analysis_df["correct"] == 0].copy()

    all_errors_path = os.path.join(
        OUTPUT_DIR,
        "dev_all_errors.csv"
    )
    errors_df.to_csv(all_errors_path, index=False)

    false_positives_df = errors_df[errors_df["error_type"] == "false_positive"].copy()
    false_negatives_df = errors_df[errors_df["error_type"] == "false_negative"].copy()

    false_positives_path = os.path.join(
        OUTPUT_DIR,
        "dev_false_positives.csv"
    )
    false_negatives_path = os.path.join(
        OUTPUT_DIR,
        "dev_false_negatives.csv"
    )

    false_positives_df.to_csv(false_positives_path, index=False)
    false_negatives_df.to_csv(false_negatives_path, index=False)

    # High-confidence wrong predictions are important because the model is confidently wrong.
    high_confidence_errors_df = errors_df.sort_values(
        by="confidence",
        ascending=False
    ).head(50)

    high_confidence_errors_path = os.path.join(
        OUTPUT_DIR,
        "dev_high_confidence_errors_top50.csv"
    )
    high_confidence_errors_df.to_csv(high_confidence_errors_path, index=False)

    # Low-margin errors are ambiguous because the model predicted close probabilities.
    low_margin_errors_df = errors_df.sort_values(
        by="margin",
        ascending=True
    ).head(50)

    low_margin_errors_path = os.path.join(
        OUTPUT_DIR,
        "dev_low_margin_errors_top50.csv"
    )
    low_margin_errors_df.to_csv(low_margin_errors_path, index=False)

    summary_rows = []

    total_examples = len(analysis_df)
    total_errors = len(errors_df)
    total_false_positives = len(false_positives_df)
    total_false_negatives = len(false_negatives_df)

    summary_rows.append({
        "metric": "total_examples",
        "value": total_examples
    })
    summary_rows.append({
        "metric": "total_errors",
        "value": total_errors
    })
    summary_rows.append({
        "metric": "false_positives",
        "value": total_false_positives
    })
    summary_rows.append({
        "metric": "false_negatives",
        "value": total_false_negatives
    })

    for lang, group_df in errors_df.groupby("lang"):
        summary_rows.append({
            "metric": f"errors_language_{lang}",
            "value": len(group_df)
        })

    for group_name, group_df in errors_df.groupby("disagreement_group"):
        summary_rows.append({
            "metric": f"errors_disagreement_{group_name}",
            "value": len(group_df)
        })

    for vote_pattern, group_df in errors_df.groupby("vote_pattern"):
        summary_rows.append({
            "metric": f"errors_vote_pattern_{vote_pattern}",
            "value": len(group_df)
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "dev_error_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n===== Error Analysis Summary =====")
    print(f"Total examples: {total_examples}")
    print(f"Total errors: {total_errors}")
    print(f"False positives: {total_false_positives}")
    print(f"False negatives: {total_false_negatives}")

    print("\nErrors by language:")
    print(errors_df["lang"].value_counts().to_string())

    print("\nErrors by disagreement group:")
    print(errors_df["disagreement_group"].value_counts().to_string())

    print("\nErrors by vote pattern:")
    print(errors_df["vote_pattern"].value_counts().to_string())

    print("\nSaved files:")
    print(all_predictions_path)
    print(all_errors_path)
    print(false_positives_path)
    print(false_negatives_path)
    print(high_confidence_errors_path)
    print(low_margin_errors_path)
    print(summary_path)


if __name__ == "__main__":
    main()