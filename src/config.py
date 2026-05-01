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
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5

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