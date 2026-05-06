import json
import pandas as pd

"""
Preprocessing utilities for EXIST 2023 Task 1.

This module handles:
1. Annotated training/dev files with labels_task1
2. Test files without labels
3. Soft-label distributions
4. Hard-label baselines
"""


TASK1_LABELS = ["NO", "YES"]


def compute_soft_label_task1(labels):
    """
    Convert annotator labels into a probability distribution.

    Example:
        ["YES", "YES", "NO"] -> {"NO": 0.3333, "YES": 0.6667}

    UNKNOWN labels are ignored if they appear.
    """
    valid_labels = [label for label in labels if label in TASK1_LABELS]

    total = len(valid_labels)

    if total == 0:
        return {
            "NO": 0.0,
            "YES": 0.0
        }

    yes_count = valid_labels.count("YES")
    no_count = valid_labels.count("NO")

    return {
        "NO": no_count / total,
        "YES": yes_count / total
    }


def load_annotated_data(path):
    """
    Load training/dev EXIST JSON files that include labels_task1.

    Returns a DataFrame with:
        id_EXIST
        lang
        text
        soft_label
        label_vector
        hard_label
        NO_value
        split
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        tweet = item["tweet"]
        labels_task1 = item["labels_task1"]

        soft_label = compute_soft_label_task1(labels_task1)

        # IMPORTANT:
        # label_vector order matches TASK1_LABELS:
        # index 0 = NO
        # index 1 = YES
        label_vector = [
            soft_label["NO"],
            soft_label["YES"]
        ]

        # hard label:
        # 0 = NO
        # 1 = YES
        hard_label = int(label_vector[1] > label_vector[0])

        rows.append({
            "id_EXIST": item.get("id_EXIST", key),
            "lang": item.get("lang", None),
            "text": tweet,
            "soft_label": soft_label,
            "label_vector": label_vector,
            "hard_label": hard_label,
            "NO_value": soft_label["NO"],
            "split": item.get("split", None)
        })

    return pd.DataFrame(rows)


def load_test_data(path):
    """
    Load EXIST test JSON files.

    Test files do not include labels.
    They only contain id_EXIST, lang, tweet, and split.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        rows.append({
            "id_EXIST": item.get("id_EXIST", key),
            "lang": item.get("lang", None),
            "text": item["tweet"],
            "split": item.get("split", None)
        })

    return pd.DataFrame(rows)


# Backward-compatible alias
# Your old train.py uses load_data().
# For now, keep it so old code does not break.
def load_data(path):
    return load_annotated_data(path)