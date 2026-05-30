import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python3 scripts/analyze_errors.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import TRAIN_PATH, DEV_PATH, TASK3_LABELS
from src.data.preprocessing import load_annotated_data_task3


OUTPUT_DIR = "outputs/analysis"


def summarize(df, split_name):
    print(f"\n===== {split_name} Task 3 Data Summary =====")
    print(f"Total valid Task 3 examples: {len(df)}")

    print("\nLanguage distribution:")
    print(df["lang"].value_counts().to_string())

    print("\nNumber of categories per example:")
    print(df["num_task3_categories"].value_counts().sort_index().to_string())

    label_counts = {}

    for label_index, label_name in enumerate(TASK3_LABELS):
        label_counts[label_name] = int(
            df["task3_binary_vector"].apply(lambda x: x[label_index]).sum()
        )

    label_counts_df = pd.DataFrame(
        label_counts.items(),
        columns=["task3_label", "example_count"]
    ).sort_values(by="example_count", ascending=False)

    print("\nTask 3 category frequency:")
    print(label_counts_df.to_string(index=False))

    return label_counts_df


def save_processed_data(df, output_path):
    df_to_save = df.copy()

    # Convert list/dict columns to strings so CSV saves cleanly.
    df_to_save["task3_soft_label"] = df_to_save["task3_soft_label"].astype(str)
    df_to_save["task3_label_vector"] = df_to_save["task3_label_vector"].astype(str)
    df_to_save["task3_binary_vector"] = df_to_save["task3_binary_vector"].astype(str)

    df_to_save.to_csv(output_path, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Task 3 training data...")
    train_df = load_annotated_data_task3(TRAIN_PATH)

    print("Loading Task 3 development data...")
    dev_df = load_annotated_data_task3(DEV_PATH)

    train_counts = summarize(train_df, "Training")
    dev_counts = summarize(dev_df, "Development")

    train_output_path = os.path.join(
        OUTPUT_DIR,
        "task3_training_processed.csv"
    )

    dev_output_path = os.path.join(
        OUTPUT_DIR,
        "task3_dev_processed.csv"
    )

    train_counts_path = os.path.join(
        OUTPUT_DIR,
        "task3_training_category_counts.csv"
    )

    dev_counts_path = os.path.join(
        OUTPUT_DIR,
        "task3_dev_category_counts.csv"
    )

    save_processed_data(train_df, train_output_path)
    save_processed_data(dev_df, dev_output_path)

    train_counts.to_csv(train_counts_path, index=False)
    dev_counts.to_csv(dev_counts_path, index=False)

    print("\nSaved files:")
    print(train_output_path)
    print(dev_output_path)
    print(train_counts_path)
    print(dev_counts_path)


if __name__ == "__main__":
    main()