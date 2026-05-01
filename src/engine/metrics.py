# src/engine/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(true_labels, predictions):
    """
    true_labels: list or numpy array
    predictions: list or numpy array
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