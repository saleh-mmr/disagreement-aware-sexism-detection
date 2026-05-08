# src/engine/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_classification_metrics(true_labels, predictions):
    """
    Accuracy: How many predictions are correct overall.
    Precision: Of everything the model predicted as sexist, how many were truly sexist.
    Recall: Of all truly sexist tweets, how many the model detected.
    F1-score: Balance between precision and recall.
    Confusion matrix:
        - true NO predicted NO
        - true NO predicted YES
        - true YES predicted NO
        - true YES predicted YES
    """

    acc = accuracy_score(true_labels, predictions)

    precision = precision_score(
        true_labels,
        predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }