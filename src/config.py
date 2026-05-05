# src/config.py

import torch

# =====================
# MODELS (ENSEMBLE)
# =====================
MODEL_NAMES = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased"
]

# =====================
# TRAINING
# =====================
"""
Meaning:
These are reasonable starting values for transformer fine-tuning.
"""
MAX_LEN = 128        # each tweet is cut or padded to 128 tokens;
BATCH_SIZE = 8       # training uses 8 examples per batch;
EPOCHS = 2           # each model trains for 2 epochs;
LR = 2e-5            # learning rate is 2e-5.

# =====================
# TASK
# =====================
NUM_CLASSES = 2

# =====================
# SYSTEM
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# PATHS
# =====================
TRAIN_PATH = "data/EXIST2023_training.json"
MODEL_OUTPUT = "outputs/models/"