import json
import os
import numpy as np

"""
Utilities for writing predictions in the official EXIST 2023 format.

For Task 1:
- hard_label must be "YES" or "NO"
- soft_label must contain probabilities for "YES" and "NO"
"""


def write_task1_predictions(ids, probabilities, output_path):
    """
    Write Task 1 predictions in official EXIST 2023 JSON format.

    ids:
        List/array of EXIST tweet IDs.

    probabilities:
        Array-like of shape (N, 2)
        index 0 = NO probability
        index 1 = YES probability

    output_path:
        Path where the prediction JSON file will be saved.
    """
    probabilities = np.asarray(probabilities)

    predictions = {}

    for tweet_id, probs in zip(ids, probabilities):
        no_prob = float(probs[0])
        yes_prob = float(probs[1])

        # Normalize defensively in case of tiny numerical issues
        total = no_prob + yes_prob
        if total > 0:
            no_prob = no_prob / total
            yes_prob = yes_prob / total

        hard_label = "YES" if yes_prob > no_prob else "NO"

        predictions[str(tweet_id)] = {
            "id_EXIST": str(tweet_id),
            "hard_label": hard_label,
            "soft_label": {
                "NO": no_prob,
                "YES": yes_prob
            }
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"Saved predictions to: {output_path}")