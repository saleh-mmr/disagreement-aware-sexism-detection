# src/engine/soft_metrics.py

import numpy as np

"""
Soft evaluation utilities.

Important:
This file does NOT implement the official EXIST ICM-Soft metric.

The official EXIST ICM-Soft metric is computed by:
    data/evaluation/exist2023evaluation.py

This file is only for internal validation monitoring during training.
"""


def soft_cross_entropy(true_probs, pred_probs):
    """
    Computes cross-entropy between true soft labels and predicted probabilities.

    true_probs: numpy array of shape (N, C)
    pred_probs: numpy array of shape (N, C)

    Lower is better.
    """
    eps = 1e-12

    true_probs = np.asarray(true_probs, dtype=np.float64)
    pred_probs = np.asarray(pred_probs, dtype=np.float64)

    pred_probs = np.clip(pred_probs, eps, 1.0)

    ce = -np.sum(true_probs * np.log(pred_probs), axis=1)

    return float(np.mean(ce))


def kl_similarity_score(true_probs, pred_probs):
    """
    Simple KL-based similarity score for internal monitoring only.

    This is NOT official EXIST ICM-Soft.

    Higher is better.
    Maximum value is 1.
    """
    eps = 1e-12

    true_probs = np.asarray(true_probs, dtype=np.float64)
    pred_probs = np.asarray(pred_probs, dtype=np.float64)

    true_probs = np.clip(true_probs, eps, 1.0)
    pred_probs = np.clip(pred_probs, eps, 1.0)

    kl_div = np.sum(
        true_probs * (np.log(true_probs) - np.log(pred_probs)),
        axis=1
    )

    score = np.exp(-kl_div)

    return float(np.mean(score))