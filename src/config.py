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
# Options:
# "soft" = train with annotator disagreement probability distributions
# "hard" = train with majority-vote hard labels
MODE = "soft"

# =====================
# TASK 1
# =====================
NUM_CLASSES = 2

# Task 1 label order:
# index 0 = NO
# index 1 = YES
TASK1_LABELS = ["NO", "YES"]

# =====================
# TASK 2
# =====================
TASK2_LABELS = ["DIRECT", "JUDGEMENTAL", "REPORTED"]
TASK2_NUM_CLASSES = 3

# =====================
# TASK 3
# =====================
TASK3_LABELS = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE"
]
TASK3_NUM_CLASSES = 5


# =====================
# SYSTEM
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATA PATHS
# =====================
TRAIN_PATH = "data/training/EXIST2023_training.json"
DEV_PATH = "data/dev/EXIST2023_dev.json"
TEST_PATH = "data/test/EXIST2023_test_clean.json"

# =====================
# EVALUATION PATHS
# =====================
EVAL_SCRIPT_PATH = "data/evaluation/exist2023evaluation.py"

DEV_TASK1_GOLD_SOFT = "data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json"
DEV_TASK1_GOLD_HARD = "data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json"

TRAIN_TASK1_GOLD_SOFT = "data/evaluation/golds/EXIST2023_training_task1_gold_soft.json"
TRAIN_TASK1_GOLD_HARD = "data/evaluation/golds/EXIST2023_training_task1_gold_hard.json"

# =====================
# OUTPUT PATHS
# =====================
MODEL_OUTPUT = "outputs/models/"
PREDICTION_OUTPUT = "outputs/predictions/"

# =====================
# TASK 2 OUTPUT PATHS
# =====================
TASK2_MODEL_OUTPUT = "outputs/models/task2/"
TASK2_PREDICTION_OUTPUT = "outputs/predictions/task2/"

# =====================
# TASK 3 OUTPUT PATHS
# =====================
TASK3_MODEL_OUTPUT = "outputs/models/task3/"
TASK3_PREDICTION_OUTPUT = "outputs/predictions/task3/"