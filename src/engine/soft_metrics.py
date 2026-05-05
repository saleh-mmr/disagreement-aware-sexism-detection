# src/engine/soft_metrics.py

import numpy as np

"""
Soft evaluation metrics for learning from disagreement.
Includes:
- Cross Entropy (soft)
- ICM-Soft (Information Closeness Measure)
"""


def soft_cross_entropy(true_probs, pred_probs):
    """
    Computes cross-entropy between two probability distributions.

    true_probs: (N, C)
    pred_probs: (N, C)
    """
    eps = 1e-12
    pred_probs = np.clip(pred_probs, eps, 1.0)

    ce = -np.sum(true_probs * np.log(pred_probs), axis=1)

    return np.mean(ce)


def icm_soft(true_probs, pred_probs):
    """
    ICM-Soft (Information Closeness Measure)

    Measures how close predicted probability distributions are
    to annotator distributions.

    Higher is better (max = 1).
    """
    eps = 1e-12
    pred_probs = np.clip(pred_probs, eps, 1.0)
    true_probs = np.clip(true_probs, eps, 1.0)

    # KL Divergence: D_KL(true || pred)
    kl_div = np.sum(true_probs * (np.log(true_probs) - np.log(pred_probs)), axis=1)

    # Convert KL → similarity score
    icm = np.exp(-kl_div)

    return np.mean(icm)