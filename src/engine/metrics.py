# src/engine/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(true_labels, predictions):
    """
    Accuracy: How many predictions are correct overall.
    Precision: Of everything the model predicted as sexist, how many were truly sexist.
    Recall: Of all truly sexist tweets, how many the model detected.
    F1-score: Balance between precision and recall. This is especially important because sexism detection may involve class imbalance.
    Confusion matrix Shows:
        - true non-sexist predicted non-sexist
        - true non-sexist predicted sexist
        - true sexist predicted non-sexist
        - true sexist predicted sexist
    """
    
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="binary")
    recall = recall_score(true_labels, predictions, average="binary")
    f1 = f1_score(true_labels, predictions, average="binary")
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }