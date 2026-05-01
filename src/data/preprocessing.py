import json
import pandas as pd

"""
Preprocessing utilities for sexism detection. 
This module converts raw JSON data and crowd-sourced labels into 
soft label distributions and hard label baselines.
"""

def compute_soft_label_task1(labels):
    total = len(labels)
    yes_count = labels.count("YES")
    no_count = labels.count("NO")

    return {
        "sexist": yes_count / total,
        "non-sexist": no_count / total
    }


def load_data(path):
    # load JSON file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for key, item in data.items():
        tweet = item["tweet"]
        labels_task1 = item["labels_task1"]

        # build soft label
        soft_label = compute_soft_label_task1(labels_task1)

        # convert to vector
        label_vector = [
            soft_label["non-sexist"],
            soft_label["sexist"]
        ]

        # convert to hard label (baseline)
        hard_label = int(label_vector[1] > label_vector[0])

        rows.append({
            "text": tweet,
            "soft_label": soft_label,
            "label_vector": label_vector,
            "hard_label": hard_label,
            "NO_value": soft_label["non-sexist"]
        })

    df = pd.DataFrame(rows)

    return df