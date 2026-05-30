import os
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python3 baselines/tfidf_logistic_regression.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.config import TRAIN_PATH, DEV_PATH
from src.data.preprocessing import load_annotated_data
from src.engine.metrics import compute_classification_metrics
from src.engine.prediction_writer import write_task1_predictions


OUTPUT_DIR = "outputs/baselines"


def save_classification_report(report_dict, output_path):
    """
    Save sklearn classification report as JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training data...")
    train_df = load_annotated_data(TRAIN_PATH)

    print("Loading development data...")
    dev_df = load_annotated_data(DEV_PATH)

    train_texts = train_df["text"].values
    dev_texts = dev_df["text"].values

    train_labels = train_df["hard_label"].values
    dev_labels = dev_df["hard_label"].values

    print("\nVectorizing text with TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)

    print(f"Train TF-IDF shape: {X_train.shape}")
    print(f"Dev TF-IDF shape: {X_dev.shape}")

    print("\nTraining Logistic Regression baseline...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    model.fit(X_train, train_labels)

    print("\nPredicting on development set...")

    dev_preds = model.predict(X_dev)
    dev_probs = model.predict_proba(X_dev)

    # LogisticRegression class order should be [0, 1], but check defensively.
    print(f"Model class order: {model.classes_}")

    if list(model.classes_) != [0, 1]:
        reordered_probs = np.zeros_like(dev_probs)
        for i, cls in enumerate(model.classes_):
            reordered_probs[:, int(cls)] = dev_probs[:, i]
        dev_probs = reordered_probs

    metrics = compute_classification_metrics(dev_labels, dev_preds)

    print("\n===== TF-IDF + Logistic Regression Results =====")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    report = classification_report(
        dev_labels,
        dev_preds,
        target_names=["NO", "YES"],
        output_dict=True,
        zero_division=0
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "tfidf_logistic_regression_classification_report.json"
    )

    save_classification_report(report, report_path)

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "tfidf_logistic_regression_dev_predictions.json"
    )

    write_task1_predictions(
        ids=dev_df["id_EXIST"].values,
        probabilities=dev_probs,
        output_path=prediction_path
    )

    error_rows = []

    for idx, row in dev_df.iterrows():
        gold = int(dev_labels[idx])
        pred = int(dev_preds[idx])
        probs = dev_probs[idx]

        if gold != pred:
            error_rows.append({
                "id_EXIST": row["id_EXIST"],
                "lang": row["lang"],
                "text": row["text"],
                "gold_label": "YES" if gold == 1 else "NO",
                "predicted_label": "YES" if pred == 1 else "NO",
                "NO_probability": float(probs[0]),
                "YES_probability": float(probs[1]),
                "confidence": float(max(probs))
            })

    errors_df = pd.DataFrame(error_rows)

    errors_path = os.path.join(
        OUTPUT_DIR,
        "tfidf_logistic_regression_errors.csv"
    )

    errors_df.to_csv(errors_path, index=False)

    print("\nSaved outputs:")
    print(report_path)
    print(prediction_path)
    print(errors_path)


if __name__ == "__main__":
    main()