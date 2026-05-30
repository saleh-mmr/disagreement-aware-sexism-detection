# src/data/preprocessing.py


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
TASK2_LABELS = ["DIRECT", "JUDGEMENTAL", "REPORTED"]
TASK3_LABELS = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE"
]


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

def compute_soft_label_task2(labels):
    """
    Convert Task 2 annotator labels into a probability distribution.

    Valid labels:
        DIRECT
        JUDGEMENTAL
        REPORTED

    UNKNOWN and '-' are ignored.
    """
    valid_labels = [label for label in labels if label in TASK2_LABELS]
    total = len(valid_labels)

    if total == 0:
        return {label: 0.0 for label in TASK2_LABELS}

    return {
        label: valid_labels.count(label) / total
        for label in TASK2_LABELS
    }

def compute_soft_label_task3(labels_task3):
    """
    Convert Task 3 annotator labels into a multi-label probability vector.

    Task 3 is multi-label because one annotator can assign multiple sexism
    categories to the same tweet.

    The output value for each category represents the fraction of annotators
    who selected that category.

    UNKNOWN and '-' are ignored.
    """
    counts = {label: 0 for label in TASK3_LABELS}
    total_annotators = len(labels_task3)

    if total_annotators == 0:
        return {label: 0.0 for label in TASK3_LABELS}

    for annotator_labels in labels_task3:
        if not isinstance(annotator_labels, list):
            continue

        unique_valid_labels = set(
            label for label in annotator_labels
            if label in TASK3_LABELS
        )

        for label in unique_valid_labels:
            counts[label] += 1

    return {
        label: counts[label] / total_annotators
        for label in TASK3_LABELS
    }

def compute_hard_label_from_distribution(distribution, labels):
    """
    Convert a probability distribution to a hard label index.

    If all probabilities are zero, return None.
    """
    values = [distribution[label] for label in labels]

    if sum(values) == 0:
        return None

    return int(max(range(len(values)), key=lambda i: values[i]))

def load_annotated_data_task2(path):
    """
    Load annotated EXIST data for Task 2.

    This function keeps only examples with valid Task 2 labels.
    It ignores '-' and UNKNOWN labels.

    Returns a DataFrame with:
        id_EXIST
        lang
        text
        task2_soft_label
        task2_label_vector
        task2_hard_label
        task2_hard_label_name
        split
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        labels_task2 = item.get("labels_task2", [])

        soft_label = compute_soft_label_task2(labels_task2)
        hard_label = compute_hard_label_from_distribution(
            soft_label,
            TASK2_LABELS
        )

        # Skip examples with no valid Task 2 label.
        if hard_label is None:
            continue

        label_vector = [
            soft_label[label]
            for label in TASK2_LABELS
        ]

        rows.append({
            "id_EXIST": item.get("id_EXIST", key),
            "lang": item.get("lang", None),
            "text": item["tweet"],
            "task2_soft_label": soft_label,
            "task2_label_vector": label_vector,
            "task2_hard_label": hard_label,
            "task2_hard_label_name": TASK2_LABELS[hard_label],
            "split": item.get("split", None)
        })

    return pd.DataFrame(rows)

def load_annotated_data_task3(path):
    """
    Load annotated EXIST data for Task 3.

    Task 3 is treated as multi-label classification.

    Returns a DataFrame with:
        id_EXIST
        lang
        text
        task3_soft_label
        task3_label_vector
        task3_binary_vector
        num_task3_categories
        split

    task3_label_vector:
        soft distribution/frequency per category

    task3_binary_vector:
        1 if category appears at least once among annotators, otherwise 0
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        labels_task3 = item.get("labels_task3", [])

        soft_label = compute_soft_label_task3(labels_task3)

        label_vector = [
            soft_label[label]
            for label in TASK3_LABELS
        ]

        binary_vector = [
            1 if value > 0 else 0
            for value in label_vector
        ]

        # Skip examples with no valid Task 3 category.
        if sum(binary_vector) == 0:
            continue

        rows.append({
            "id_EXIST": item.get("id_EXIST", key),
            "lang": item.get("lang", None),
            "text": item["tweet"],
            "task3_soft_label": soft_label,
            "task3_label_vector": label_vector,
            "task3_binary_vector": binary_vector,
            "num_task3_categories": sum(binary_vector),
            "split": item.get("split", None)
        })

    return pd.DataFrame(rows)

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


def load_data(path):
    return load_annotated_data(path)