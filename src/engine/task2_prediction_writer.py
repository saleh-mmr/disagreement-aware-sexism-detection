import json
import os
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python scripts/benchmark_runtime.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import TASK2_LABELS


def write_task2_predictions(ids, probabilities, output_path):
    """
    Write Task 2 predictions in a simple JSON format.

    probabilities:
        shape = (N, 3)

    Label order:
        index 0 = DIRECT
        index 1 = JUDGEMENTAL
        index 2 = REPORTED
    """
    probabilities = np.asarray(probabilities)

    predictions = {}

    for tweet_id, probs in zip(ids, probabilities):
        probs = np.asarray(probs, dtype=float)

        total = probs.sum()
        if total > 0:
            probs = probs / total

        hard_index = int(np.argmax(probs))
        hard_label = TASK2_LABELS[hard_index]

        predictions[str(tweet_id)] = {
            "id_EXIST": str(tweet_id),
            "hard_label": hard_label,
            "soft_label": {
                TASK2_LABELS[i]: float(probs[i])
                for i in range(len(TASK2_LABELS))
            }
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved Task 2 predictions to: {output_path}")