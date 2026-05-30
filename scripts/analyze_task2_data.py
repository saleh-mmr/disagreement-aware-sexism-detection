import os
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python scripts/benchmark_runtime.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import TRAIN_PATH, DEV_PATH
from src.data.preprocessing import load_annotated_data_task2


OUTPUT_DIR = "outputs/analysis"


def summarize(df, split_name):
    print(f"\n===== {split_name} Task 2 Data Summary =====")
    print(f"Total valid Task 2 examples: {len(df)}")

    print("\nLabel distribution:")
    print(df["task2_hard_label_name"].value_counts().to_string())

    print("\nLanguage distribution:")
    print(df["lang"].value_counts().to_string())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Task 2 training data...")
    train_df = load_annotated_data_task2(TRAIN_PATH)

    print("Loading Task 2 development data...")
    dev_df = load_annotated_data_task2(DEV_PATH)

    summarize(train_df, "Training")
    summarize(dev_df, "Development")

    train_output_path = os.path.join(
        OUTPUT_DIR,
        "task2_training_processed.csv"
    )

    dev_output_path = os.path.join(
        OUTPUT_DIR,
        "task2_dev_processed.csv"
    )

    train_df.to_csv(train_output_path, index=False)
    dev_df.to_csv(dev_output_path, index=False)

    print("\nSaved files:")
    print(train_output_path)
    print(dev_output_path)


if __name__ == "__main__":
    main()