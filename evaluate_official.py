# evaluate_official.py


import subprocess
from src.config import (
    EVAL_SCRIPT_PATH,
    DEV_TASK1_GOLD_SOFT,
    DEV_TASK1_GOLD_HARD,
    PREDICTION_OUTPUT
)

"""
Runs the official EXIST 2023 evaluator for Task 1.

Before running this file, you should already have:
outputs/predictions/dev_task1_ensemble_predictions.json
"""


def main():
    prediction_path = PREDICTION_OUTPUT + "dev_task1_ensemble_predictions.json"

    command = [
        "python",
        EVAL_SCRIPT_PATH,
        "-p", prediction_path,
        "-g", DEV_TASK1_GOLD_SOFT,
        "-e", DEV_TASK1_GOLD_HARD,
        "-t", "task1"
    ]

    print("Running official EXIST evaluation...")
    print(" ".join(command))

    result = subprocess.run(
        command,
        text=True,
        capture_output=True
    )

    print("\n===== OFFICIAL EVALUATION OUTPUT =====")
    print(result.stdout)

    if result.stderr:
        print("\n===== ERRORS / WARNINGS =====")
        print(result.stderr)


if __name__ == "__main__":
    main()