import os
import json
from collections import Counter
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python scripts/benchmark_runtime.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_PATH


OUTPUT_DIR = "outputs/analysis"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_annotation_rows(data):
    """
    Convert tweet-level EXIST examples into annotation-level rows.

    Each tweet has multiple annotators, so this creates one row per annotator.
    """
    rows = []

    for item_id, item in data.items():
        labels_task1 = item.get("labels_task1", [])
        labels_task2 = item.get("labels_task2", [])
        labels_task3 = item.get("labels_task3", [])
        gender_annotators = item.get("gender_annotators", [])
        age_annotators = item.get("age_annotators", [])

        for i in range(len(labels_task1)):
            task3_labels = labels_task3[i] if i < len(labels_task3) else ["-"]

            rows.append({
                "id_EXIST": item.get("id_EXIST", item_id),
                "lang": item.get("lang"),
                "split": item.get("split"),
                "annotator_index": i,
                "annotator_gender": gender_annotators[i] if i < len(gender_annotators) else None,
                "annotator_age": age_annotators[i] if i < len(age_annotators) else None,
                "task1_label": labels_task1[i] if i < len(labels_task1) else None,
                "task2_label": labels_task2[i] if i < len(labels_task2) else None,
                "task3_labels": task3_labels
            })

    return pd.DataFrame(rows)


def summarize_task1_by_group(df, group_column):
    rows = []

    for group_value, group_df in df.groupby(group_column):
        total = len(group_df)
        yes_count = int((group_df["task1_label"] == "YES").sum())
        no_count = int((group_df["task1_label"] == "NO").sum())

        rows.append({
            group_column: group_value,
            "total_annotations": total,
            "YES_count": yes_count,
            "NO_count": no_count,
            "YES_rate": yes_count / total if total > 0 else 0.0,
            "NO_rate": no_count / total if total > 0 else 0.0
        })

    return pd.DataFrame(rows)


def summarize_task2_by_group(df, group_column):
    rows = []

    for group_value, group_df in df.groupby(group_column):
        labels = [
            label for label in group_df["task2_label"].tolist()
            if label not in [None, "-"]
        ]

        counter = Counter(labels)
        total_valid = sum(counter.values())

        for label, count in counter.items():
            rows.append({
                group_column: group_value,
                "task2_label": label,
                "count": count,
                "rate_within_group": count / total_valid if total_valid > 0 else 0.0
            })

    result = pd.DataFrame(rows)

    if len(result) > 0:
        result = result.sort_values(
            by=[group_column, "count"],
            ascending=[True, False]
        )

    return result


def summarize_task3_by_group(df, group_column):
    rows = []

    for group_value, group_df in df.groupby(group_column):
        all_labels = []

        for labels in group_df["task3_labels"]:
            if isinstance(labels, list):
                for label in labels:
                    if label != "-":
                        all_labels.append(label)

        counter = Counter(all_labels)
        total_valid = sum(counter.values())

        for label, count in counter.items():
            rows.append({
                group_column: group_value,
                "task3_label": label,
                "count": count,
                "rate_within_group": count / total_valid if total_valid > 0 else 0.0
            })

    result = pd.DataFrame(rows)

    if len(result) > 0:
        result = result.sort_values(
            by=[group_column, "count"],
            ascending=[True, False]
        )

    return result


def compute_item_level_vote_gaps(data):
    """
    Compute per-tweet YES-rate differences between annotator demographic groups.
    """
    rows = []

    for item_id, item in data.items():
        labels_task1 = item.get("labels_task1", [])
        genders = item.get("gender_annotators", [])
        ages = item.get("age_annotators", [])

        female_votes = []
        male_votes = []

        age_votes = {
            "18-22": [],
            "23-45": [],
            "46+": []
        }

        for label, gender, age in zip(labels_task1, genders, ages):
            vote = 1 if label == "YES" else 0

            if gender == "F":
                female_votes.append(vote)
            elif gender == "M":
                male_votes.append(vote)

            if age in age_votes:
                age_votes[age].append(vote)

        female_yes_rate = sum(female_votes) / len(female_votes) if female_votes else None
        male_yes_rate = sum(male_votes) / len(male_votes) if male_votes else None

        age_rates = {}
        for age_group, votes in age_votes.items():
            age_rates[age_group] = sum(votes) / len(votes) if votes else None

        if None not in age_rates.values():
            age_range = max(age_rates.values()) - min(age_rates.values())
        else:
            age_range = None

        if female_yes_rate is not None and male_yes_rate is not None:
            gender_gap = abs(female_yes_rate - male_yes_rate)
        else:
            gender_gap = None

        rows.append({
            "id_EXIST": item.get("id_EXIST", item_id),
            "lang": item.get("lang"),
            "text": item.get("tweet", ""),
            "female_yes_rate": female_yes_rate,
            "male_yes_rate": male_yes_rate,
            "gender_yes_rate_gap_abs": gender_gap,
            "age_18_22_yes_rate": age_rates["18-22"],
            "age_23_45_yes_rate": age_rates["23-45"],
            "age_46_plus_yes_rate": age_rates["46+"],
            "age_yes_rate_range": age_range
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading training dataset...")
    data = load_json(TRAIN_PATH)

    print("Creating annotation-level dataframe...")
    annotation_df = flatten_annotation_rows(data)

    annotation_path = os.path.join(
        OUTPUT_DIR,
        "annotator_level_annotations.csv"
    )
    annotation_df.to_csv(annotation_path, index=False)

    task1_by_gender = summarize_task1_by_group(annotation_df, "annotator_gender")
    task1_by_age = summarize_task1_by_group(annotation_df, "annotator_age")

    task2_by_gender = summarize_task2_by_group(annotation_df, "annotator_gender")
    task2_by_age = summarize_task2_by_group(annotation_df, "annotator_age")

    task3_by_gender = summarize_task3_by_group(annotation_df, "annotator_gender")
    task3_by_age = summarize_task3_by_group(annotation_df, "annotator_age")

    item_gap_df = compute_item_level_vote_gaps(data)

    paths = {
        "task1_by_gender": os.path.join(OUTPUT_DIR, "task1_vote_rate_by_annotator_gender.csv"),
        "task1_by_age": os.path.join(OUTPUT_DIR, "task1_vote_rate_by_annotator_age.csv"),
        "task2_by_gender": os.path.join(OUTPUT_DIR, "task2_distribution_by_annotator_gender.csv"),
        "task2_by_age": os.path.join(OUTPUT_DIR, "task2_distribution_by_annotator_age.csv"),
        "task3_by_gender": os.path.join(OUTPUT_DIR, "task3_distribution_by_annotator_gender.csv"),
        "task3_by_age": os.path.join(OUTPUT_DIR, "task3_distribution_by_annotator_age.csv"),
        "item_gaps": os.path.join(OUTPUT_DIR, "item_level_annotator_demographic_vote_gaps.csv")
    }

    task1_by_gender.to_csv(paths["task1_by_gender"], index=False)
    task1_by_age.to_csv(paths["task1_by_age"], index=False)
    task2_by_gender.to_csv(paths["task2_by_gender"], index=False)
    task2_by_age.to_csv(paths["task2_by_age"], index=False)
    task3_by_gender.to_csv(paths["task3_by_gender"], index=False)
    task3_by_age.to_csv(paths["task3_by_age"], index=False)
    item_gap_df.to_csv(paths["item_gaps"], index=False)

    print("\n===== Task 1 Vote Rate by Annotator Gender =====")
    print(task1_by_gender.to_string(index=False))

    print("\n===== Task 1 Vote Rate by Annotator Age =====")
    print(task1_by_age.to_string(index=False))

    print("\n===== Task 2 Distribution by Annotator Gender =====")
    print(task2_by_gender.to_string(index=False))

    print("\n===== Task 3 Distribution by Annotator Gender =====")
    print(task3_by_gender.to_string(index=False))

    print("\nSaved files:")
    print(annotation_path)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()