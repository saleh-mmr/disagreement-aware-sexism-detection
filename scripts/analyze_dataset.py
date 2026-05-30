import json
import os
import sys
from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_PATH


TASK1_LABELS = ["NO", "YES"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def majority_label(labels):
    """
    Returns majority label for Task 1.
    If YES and NO are tied, returns TIE.
    """
    yes_count = labels.count("YES")
    no_count = labels.count("NO")

    if yes_count > no_count:
        return "YES"
    elif no_count > yes_count:
        return "NO"
    else:
        return "TIE"


def disagreement_level(labels):
    """
    Measures annotator disagreement for Task 1.

    Since each example has 6 annotators, possible YES counts are:
    0, 1, 2, 3, 4, 5, 6.

    0 or 6 means full agreement.
    3 means maximum disagreement.
    """
    yes_count = labels.count("YES")
    no_count = labels.count("NO")

    if yes_count == 0 or no_count == 0:
        return "full_agreement"
    elif yes_count == 3 and no_count == 3:
        return "high_disagreement"
    else:
        return "partial_disagreement"


def flatten_task3(labels_task3):
    """
    labels_task3 is a list of lists because one annotator can assign
    multiple sexism categories.

    Example:
        [
            ["OBJECTIFICATION"],
            ["OBJECTIFICATION", "SEXUAL-VIOLENCE"],
            ["-"]
        ]

    This function flattens it into:
        ["OBJECTIFICATION", "OBJECTIFICATION", "SEXUAL-VIOLENCE"]
    """
    flattened = []

    for annotator_labels in labels_task3:
        for label in annotator_labels:
            if label != "-":
                flattened.append(label)

    return flattened


def analyze_dataset(data):
    total_examples = len(data)

    language_counter = Counter()
    task1_majority_counter = Counter()
    task1_vote_distribution = Counter()
    disagreement_counter = Counter()
    task2_counter = Counter()
    task3_counter = Counter()
    annotator_gender_counter = Counter()
    annotator_age_counter = Counter()

    rows = []

    for item_id, item in data.items():
        lang = item.get("lang")
        labels_task1 = item.get("labels_task1", [])
        labels_task2 = item.get("labels_task2", [])
        labels_task3 = item.get("labels_task3", [])
        gender_annotators = item.get("gender_annotators", [])
        age_annotators = item.get("age_annotators", [])

        yes_count = labels_task1.count("YES")
        no_count = labels_task1.count("NO")

        maj_label = majority_label(labels_task1)
        disagreement = disagreement_level(labels_task1)

        language_counter[lang] += 1
        task1_majority_counter[maj_label] += 1
        task1_vote_distribution[f"{yes_count}_YES_{no_count}_NO"] += 1
        disagreement_counter[disagreement] += 1

        for label in labels_task2:
            if label != "-":
                task2_counter[label] += 1

        for label in flatten_task3(labels_task3):
            task3_counter[label] += 1

        for gender in gender_annotators:
            annotator_gender_counter[gender] += 1

        for age in age_annotators:
            annotator_age_counter[age] += 1

        rows.append({
            "id_EXIST": item.get("id_EXIST", item_id),
            "lang": lang,
            "text": item.get("tweet", ""),
            "yes_votes": yes_count,
            "no_votes": no_count,
            "majority_label": maj_label,
            "disagreement_level": disagreement,
            "split": item.get("split")
        })

    analysis_df = pd.DataFrame(rows)

    return {
        "total_examples": total_examples,
        "language_distribution": language_counter,
        "task1_majority_distribution": task1_majority_counter,
        "task1_vote_distribution": task1_vote_distribution,
        "disagreement_distribution": disagreement_counter,
        "task2_distribution": task2_counter,
        "task3_distribution": task3_counter,
        "annotator_gender_distribution": annotator_gender_counter,
        "annotator_age_distribution": annotator_age_counter,
        "analysis_df": analysis_df
    }


def counter_to_dataframe(counter, name_column, count_column):
    df = pd.DataFrame(counter.items(), columns=[name_column, count_column])
    df = df.sort_values(by=count_column, ascending=False)
    return df


def save_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results["analysis_df"].to_csv(
        os.path.join(output_dir, "dataset_examples_analysis.csv"),
        index=False
    )

    counter_to_dataframe(
        results["language_distribution"],
        "language",
        "count"
    ).to_csv(os.path.join(output_dir, "language_distribution.csv"), index=False)

    counter_to_dataframe(
        results["task1_majority_distribution"],
        "majority_label",
        "count"
    ).to_csv(os.path.join(output_dir, "task1_majority_distribution.csv"), index=False)

    counter_to_dataframe(
        results["task1_vote_distribution"],
        "vote_distribution",
        "count"
    ).to_csv(os.path.join(output_dir, "task1_vote_distribution.csv"), index=False)

    counter_to_dataframe(
        results["disagreement_distribution"],
        "disagreement_level",
        "count"
    ).to_csv(os.path.join(output_dir, "disagreement_distribution.csv"), index=False)

    counter_to_dataframe(
        results["task2_distribution"],
        "task2_label",
        "count"
    ).to_csv(os.path.join(output_dir, "task2_distribution.csv"), index=False)

    counter_to_dataframe(
        results["task3_distribution"],
        "task3_label",
        "count"
    ).to_csv(os.path.join(output_dir, "task3_distribution.csv"), index=False)

    counter_to_dataframe(
        results["annotator_gender_distribution"],
        "annotator_gender",
        "count"
    ).to_csv(os.path.join(output_dir, "annotator_gender_distribution.csv"), index=False)

    counter_to_dataframe(
        results["annotator_age_distribution"],
        "annotator_age",
        "count"
    ).to_csv(os.path.join(output_dir, "annotator_age_distribution.csv"), index=False)


def print_counter(title, counter):
    print(f"\n===== {title} =====")
    for key, value in counter.most_common():
        print(f"{key}: {value}")


def main():
    output_dir = "outputs/analysis"

    print("Loading training dataset...")
    data = load_json(TRAIN_PATH)

    print("Analyzing dataset...")
    results = analyze_dataset(data)

    print("\n==============================")
    print("DATASET ANALYSIS SUMMARY")
    print("==============================")

    print(f"\nTotal examples: {results['total_examples']}")

    print_counter("Language Distribution", results["language_distribution"])
    print_counter("Task 1 Majority Label Distribution", results["task1_majority_distribution"])
    print_counter("Task 1 Vote Distribution", results["task1_vote_distribution"])
    print_counter("Disagreement Distribution", results["disagreement_distribution"])
    print_counter("Task 2 Label Distribution", results["task2_distribution"])
    print_counter("Task 3 Label Distribution", results["task3_distribution"])
    print_counter("Annotator Gender Distribution", results["annotator_gender_distribution"])
    print_counter("Annotator Age Distribution", results["annotator_age_distribution"])

    save_outputs(results, output_dir)

    print("\nSaved analysis files to:")
    print(output_dir)


if __name__ == "__main__":
    main()