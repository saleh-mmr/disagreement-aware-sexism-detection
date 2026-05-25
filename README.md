# Learning from Disagreement for Multilingual Sexism Detection

A complete **EXIST 2023 Task 1** pipeline for multilingual sexism identification in English and Spanish social media posts.

This project trains transformer-based classifiers for binary sexism detection while preserving annotator disagreement through soft labels. Instead of forcing multiple human annotations into one majority label only, the system can learn from the full probability distribution of annotator opinions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task Definition](#task-definition)
3. [Main Idea: Learning from Disagreement](#main-idea-learning-from-disagreement)
4. [Main Features](#main-features)
5. [Tech Stack](#tech-stack)
6. [Repository Structure](#repository-structure)
7. [Dataset Structure](#dataset-structure)
8. [Data Preprocessing](#data-preprocessing)
9. [PyTorch Dataset](#pytorch-dataset)
10. [Model Architecture](#model-architecture)
11. [Loss Functions](#loss-functions)
12. [Training Pipeline](#training-pipeline)
13. [Evaluation Pipeline](#evaluation-pipeline)
14. [Prediction and Ensemble Inference](#prediction-and-ensemble-inference)
15. [Official EXIST Prediction Format](#official-exist-prediction-format)
16. [Configuration](#configuration)
17. [How to Run the Project](#how-to-run-the-project)
18. [Outputs](#outputs)
19. [File-by-File Explanation](#file-by-file-explanation)
20. [Implementation Notes](#implementation-notes)
21. [Current Scope](#current-scope)
22. [Possible Future Improvements](#possible-future-improvements)
23. [Summary](#summary)

---

## Project Overview

This project implements a multilingual sexism detection system based on the **EXIST 2023** dataset.

The system focuses on **Task 1: Sexism Identification**, where each social media post must be classified as either sexist or non-sexist.

The project supports both English and Spanish posts and uses pretrained multilingual transformer models to learn representations across both languages.

The final system can:

1. Load the official EXIST training, development, and test datasets.
2. Convert annotator labels into soft probability distributions.
3. Train multilingual transformer models.
4. Support both hard-label and soft-label training.
5. Use KL Divergence loss for soft-label learning.
6. Combine multiple model predictions using ensemble averaging.
7. Save predictions in the official EXIST Task 1 JSON format.
8. Run the official EXIST evaluator on development predictions.
9. Generate final prediction files for the test set.

---

## Task Definition

The project currently implements:

```text
EXIST 2023 Task 1: Sexism Identification
```

This is a binary classification task:

```text
NO  = non-sexist
YES = sexist
```

Internally, the project uses the following class order:

```python
TASK1_LABELS = ["NO", "YES"]
```

Therefore:

```text
index 0 = NO
index 1 = YES
```

This order is used consistently in preprocessing, training, prediction, ensembling, and official JSON output writing.

---

## Main Idea: Learning from Disagreement

Traditional classification often converts multiple human annotations into a single majority-vote label.

Example:

```text
Annotator labels: ["YES", "YES", "NO"]
Majority label:   "YES"
```

This removes useful information because the model no longer sees that one annotator disagreed.

This project uses a **learning from disagreement** approach. Instead of training only on a single hard label, it converts annotator votes into a soft probability distribution.

Example:

```text
Annotator labels: ["YES", "YES", "NO"]

Soft label:
NO  = 1 / 3 = 0.3333
YES = 2 / 3 = 0.6667
```

The model can therefore learn:

```json
{
  "NO": 0.3333,
  "YES": 0.6667
}
```

instead of only:

```text
YES
```

This is important because sexism detection can be subjective. Some posts are clear, while others are ambiguous or context-dependent. Soft labels allow the model to represent uncertainty and disagreement instead of treating every example as completely certain.

---

## Main Features

- Multilingual sexism detection for English and Spanish text.
- EXIST 2023 Task 1 support.
- Soft-label creation from annotator disagreement.
- Hard-label baseline support.
- Transformer-based classification using:
  - `xlm-roberta-base`
  - `bert-base-multilingual-cased`
- Mean pooling and max pooling over transformer token embeddings.
- Dropout-based regularization.
- KL Divergence loss for soft-label learning.
- Cross Entropy loss for hard-label training.
- Ensemble averaging / soft voting across models.
- Official EXIST-format JSON prediction writing.
- Official development-set evaluation using the EXIST evaluator.
- Test-set inference without labels.

---

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Transformer library | HuggingFace Transformers |
| Main models | `xlm-roberta-base`, `bert-base-multilingual-cased` |
| Training modes | `soft` and `hard` |
| Soft-label loss | KL Divergence |
| Hard-label loss | Cross Entropy |
| Ensemble method | Mean probability averaging |
| Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Internal soft metrics | Soft Cross Entropy, KL Similarity |
| Official evaluation | EXIST 2023 evaluation script |
| Supported languages | English and Spanish |

---

## Repository Structure

The current project structure is:

```text
project-root/
├── README.md
├── train.py
├── evaluate_official.py
├── predict_test.py
├── requirements.txt
├── data/
│   ├── training/
│   │   └── EXIST2023_training.json
│   ├── dev/
│   │   └── EXIST2023_dev.json
│   ├── test/
│   │   └── EXIST2023_test_clean.json
│   └── evaluation/
│       ├── exist2023evaluation.py
│       ├── golds/
│       │   ├── EXIST2023_dev_task1_gold_soft.json
│       │   ├── EXIST2023_dev_task1_gold_hard.json
│       │   ├── EXIST2023_training_task1_gold_soft.json
│       │   └── EXIST2023_training_task1_gold_hard.json
│       └── baselines/
├── outputs/
│   ├── models/
│   └── predictions/
└── src/
    ├── config.py
    ├── data/
    │   ├── dataset.py
    │   └── preprocessing.py
    ├── engine/
    │   ├── evaluator.py
    │   ├── metrics.py
    │   ├── predictor.py
    │   ├── prediction_writer.py
    │   ├── soft_metrics.py
    │   └── trainer.py
    ├── losses/
    │   └── losses.py
    └── models/
        └── transformer.py
```

---

## Dataset Structure

The project uses three main EXIST data splits.

### Training set

```text
data/training/EXIST2023_training.json
```

This file is used to train the models. It contains tweets and annotator labels for Task 1.

### Development set

```text
data/dev/EXIST2023_dev.json
```

This file is used for validation and development-set prediction.

Development predictions are compared against the official gold files:

```text
data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json
data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json
```

### Test set

```text
data/test/EXIST2023_test_clean.json
```

The test set does not include labels. It is used only for final prediction generation.

The produced test prediction file is:

```text
outputs/predictions/test_task1_ensemble_predictions.json
```

---

## EXIST JSON Format

The raw EXIST files are JSON objects. Each key corresponds to one tweet/post.

Example training or development item:

```json
{
  "tweet_id_1": {
    "id_EXIST": "unique_id",
    "lang": "en",
    "tweet": "The actual text content here...",
    "labels_task1": ["YES", "YES", "NO"],
    "split": "train"
  }
}
```

Important fields:

| Field | Meaning |
|---|---|
| `id_EXIST` | Unique EXIST identifier |
| `lang` | Language of the post, usually English or Spanish |
| `tweet` | Text content to classify |
| `labels_task1` | Multiple annotator labels for Task 1 |
| `split` | Dataset split |

Test files contain no labels and are loaded separately.

---

## Data Preprocessing

Preprocessing is implemented in:

```text
src/data/preprocessing.py
```

This module handles:

1. Annotated training/development files with `labels_task1`.
2. Test files without labels.
3. Soft-label distributions.
4. Hard-label baselines.

### Label set

The project defines:

```python
TASK1_LABELS = ["NO", "YES"]
```

Only labels in this list are used. If an unknown label appears, it is ignored during soft-label computation.

### Soft-label creation

Function:

```python
compute_soft_label_task1(labels)
```

Example input:

```python
["YES", "YES", "NO"]
```

Output:

```python
{
    "NO": 0.3333,
    "YES": 0.6667
}
```

If no valid labels are found, the function returns:

```python
{
    "NO": 0.0,
    "YES": 0.0
}
```

### Label vector

The soft-label dictionary is converted into a fixed-order vector:

```python
label_vector = [
    soft_label["NO"],
    soft_label["YES"]
]
```

Example:

```python
label_vector = [0.3333, 0.6667]
```

This vector is the target used during soft-label training.

### Hard-label creation

A hard label is also created for hard-label training and metric calculation:

```python
hard_label = int(label_vector[1] > label_vector[0])
```

This means:

```text
If YES probability > NO probability, hard_label = 1
Otherwise, hard_label = 0
```

Examples:

```text
[0.3333, 0.6667] → 1 → YES
[0.8000, 0.2000] → 0 → NO
```

### Annotated data loading

Function:

```python
load_annotated_data(path)
```

This loads training or development JSON files that include `labels_task1`.

It returns a Pandas DataFrame with:

| Column | Description |
|---|---|
| `id_EXIST` | Tweet/post identifier |
| `lang` | Language |
| `text` | Tweet text |
| `soft_label` | Dictionary with `NO` and `YES` probabilities |
| `label_vector` | List in `[NO, YES]` order |
| `hard_label` | Integer class label |
| `NO_value` | Probability of `NO` |
| `split` | Dataset split |

### Test data loading

Function:

```python
load_test_data(path)
```

This loads EXIST test files without labels.

It returns:

| Column | Description |
|---|---|
| `id_EXIST` | Tweet/post identifier |
| `lang` | Language |
| `text` | Tweet text |
| `split` | Dataset split |

### Backward compatibility

The preprocessing file also defines:

```python
load_data(path)
```

This is an alias for:

```python
load_annotated_data(path)
```

It exists to keep older code compatible.

### Preprocessing flow

```text
Raw EXIST JSON
    ↓
load_annotated_data()
    ↓
Read tweet and labels_task1
    ↓
compute_soft_label_task1()
    ↓
Create label_vector = [NO_probability, YES_probability]
    ↓
Create hard_label
    ↓
Build DataFrame
    ↓
Pass text and labels into SexismDataset
```

---

## PyTorch Dataset

The dataset class is implemented in:

```text
src/data/dataset.py
```

Main class:

```python
SexismDataset
```

The dataset supports both:

1. Training/development data with labels.
2. Test data without labels.

### Responsibilities

The dataset:

1. Receives texts, labels, tokenizer, and maximum sequence length.
2. Tokenizes each text using the HuggingFace tokenizer.
3. Pads/truncates each sequence to `max_len`.
4. Returns tensors for PyTorch training or inference.

### Tokenization

Each text is encoded with:

```python
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)
```

Each item always contains:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor
}
```

If labels are available, the item also contains:

```python
"labels": tensor
```

### Label dtype handling

The dataset automatically chooses the correct label type:

```text
soft label list → torch.float
hard label int  → torch.long
```

This is important because the training and evaluation code use the label dtype to decide whether to use soft-label loss or hard-label loss.

Example soft-label item:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor,
    "labels": torch.tensor([0.3333, 0.6667], dtype=torch.float)
}
```

Example hard-label item:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor,
    "labels": torch.tensor(1, dtype=torch.long)
}
```

For test prediction, labels are set to `None`, so the dataset returns only `input_ids` and `attention_mask`.

---

## Model Architecture

The model is implemented in:

```text
src/models/transformer.py
```

Main class:

```python
TransformerModel
```

The model loads a pretrained multilingual transformer encoder with:

```python
AutoModel.from_pretrained(model_name)
```

The configured models are:

```python
MODEL_NAMES = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased"
]
```

These models are suitable because the dataset contains English and Spanish tweets.

### Architecture flow

```text
input_ids + attention_mask
        ↓
Pretrained transformer encoder
        ↓
last_hidden_state
        ↓
Mean pooling
        ↓
Max pooling
        ↓
Concatenate mean + max pooled vectors
        ↓
Dropout
        ↓
Linear classifier
        ↓
Logits for [NO, YES]
```

### Transformer encoder output

The encoder returns token-level contextual embeddings:

```text
hidden_states = [batch_size, sequence_length, hidden_size]
```

For the configured base models, the hidden size is usually:

```text
hidden_size = 768
```

With the default configuration:

```text
batch_size = 8
max_len = 128
hidden_size = 768
```

the encoder output shape is typically:

```text
[8, 128, 768]
```

### Mean pooling

Mean pooling averages token embeddings while ignoring padding tokens.

Purpose:

```text
Capture the overall meaning of the tweet.
```

Implementation idea:

```python
mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
summed = torch.sum(hidden_states * mask, dim=1)
counts = torch.clamp(mask.sum(dim=1), min=1e-9)
mean_pool = summed / counts
```

### Max pooling

Max pooling takes the strongest feature value across valid tokens.

Purpose:

```text
Capture the most salient signal in the tweet.
```

Padding tokens are ignored by replacing their values with a very small number:

```python
masked_hidden_states = hidden_states.masked_fill(~mask, -1e9)
max_pool = torch.max(masked_hidden_states, dim=1)[0]
```

This avoids modifying the original hidden states in-place, which is safer for PyTorch autograd.

### Pooling concatenation

Both pooled vectors are concatenated:

```python
pooled = torch.cat((mean_pool, max_pool), dim=1)
```

If the hidden size is `768`, then:

```text
mean_pool = [batch_size, 768]
max_pool  = [batch_size, 768]
pooled    = [batch_size, 1536]
```

This gives the classifier a richer sentence representation than using only one pooling strategy.

### Classification head

The classification head is:

```python
self.dropout = nn.Dropout(0.3)
self.classifier = nn.Linear(hidden_size * 2, num_classes)
```

For the current task:

```text
num_classes = 2
```

The model outputs logits:

```text
[NO_score, YES_score]
```

These are raw scores, not probabilities. During prediction and evaluation, they are converted to probabilities using softmax.

---

## Loss Functions

Loss functions are implemented in:

```text
src/losses/losses.py
```

The project supports two modes:

```python
MODE = "soft"
```

or:

```python
MODE = "hard"
```

### Hard-label training

Hard-label training uses:

```python
torch.nn.CrossEntropyLoss()
```

This is used when labels are integer class IDs:

```text
0 = NO
1 = YES
```

Cross Entropy expects raw logits and integer labels.

### Soft-label training

Soft-label training uses the custom class:

```python
SoftLabelLoss
```

This wraps:

```python
torch.nn.KLDivLoss(reduction="batchmean")
```

The forward pass is:

```python
log_probs = torch.log_softmax(logits, dim=1)
loss = KLDivLoss(log_probs, targets)
```

The model prediction is converted to log-probabilities, while the target is the annotator disagreement distribution.

Example:

```text
Target distribution:    [0.3333, 0.6667]
Predicted distribution: [0.2000, 0.8000]
```

KL Divergence penalizes the difference between the predicted distribution and the annotator distribution.

This means the model learns:

1. Which label is more likely.
2. How uncertain or controversial the example is.

---

## Training Pipeline

The main training script is:

```text
train.py
```

Run it with:

```bash
python train.py
```

### Training flow

```text
Create output folders
        ↓
Load training data
        ↓
Load development data
        ↓
Select labels based on MODE
        ↓
For each model in MODEL_NAMES:
    Load tokenizer
    Build SexismDataset
    Build DataLoader
    Initialize TransformerModel
    Train for EPOCHS
    Evaluate after each epoch
    Save best checkpoint by validation loss
    Predict probabilities on dev set
        ↓
Average probabilities from all models
        ↓
Write official dev prediction JSON
        ↓
Compute ensemble classification metrics
```

### Label selection

Inside `train.py`, the labels are selected according to `MODE`.

Soft mode:

```python
train_labels = train_df["label_vector"].values
dev_labels = dev_df["label_vector"].values
```

Hard mode:

```python
train_labels = train_df["hard_label"].values
dev_labels = dev_df["hard_label"].values
```

### Training one model

Function:

```python
train_single_model(model_name, train_loader, val_loader)
```

This function:

1. Initializes the transformer model.
2. Moves it to the configured device.
3. Creates an AdamW optimizer.
4. Trains for the configured number of epochs.
5. Evaluates on the development set after each epoch.
6. Saves the best checkpoint when validation loss improves.

The optimizer is:

```python
torch.optim.AdamW(model.parameters(), lr=LR)
```

### Training one epoch

Function:

```python
train_one_epoch(model, data_loader, optimizer, device)
```

This function:

1. Sets the model to training mode with `model.train()`.
2. Iterates over batches.
3. Moves tensors to CPU/GPU.
4. Runs a forward pass.
5. Selects the correct loss function automatically.
6. Backpropagates the loss.
7. Updates model parameters.
8. Tracks average loss and accuracy.

### Automatic loss selection

The trainer chooses loss based on label dtype:

```python
if labels.dtype == torch.long:
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = SoftLabelLoss()
```

This allows the same training code to support both hard-label and soft-label training.

### Monitoring accuracy in soft mode

In soft mode, accuracy is monitored by converting the soft label into a hard class:

```python
hard_labels = torch.argmax(labels, dim=1)
```

Example:

```text
[0.3333, 0.6667] → YES
[0.8000, 0.2000] → NO
```

This accuracy is useful for monitoring but does not fully represent the learning-from-disagreement objective.

### Checkpointing

The best model is saved when validation loss improves.

Checkpoint path format:

```text
outputs/models/{model_name}_{MODE}.pt
```

Because model names may contain `/`, the slash is replaced:

```python
model_name.replace("/", "_")
```

Examples for soft mode:

```text
outputs/models/xlm-roberta-base_soft.pt
outputs/models/bert-base-multilingual-cased_soft.pt
```

Examples for hard mode:

```text
outputs/models/xlm-roberta-base_hard.pt
outputs/models/bert-base-multilingual-cased_hard.pt
```

---

## Evaluation Pipeline

Evaluation is implemented in:

```text
src/engine/evaluator.py
```

Main function:

```python
evaluate(model, data_loader, device)
```

This evaluates the model without updating weights.

It uses:

```python
model.eval()
torch.no_grad()
```

This disables dropout and prevents gradient computation.

### Evaluation flow

```text
Set model to evaluation mode
        ↓
Disable gradient calculation
        ↓
For each batch:
    Move tensors to device
    Run forward pass
    Convert logits to probabilities
    Compute loss
    Convert predictions to hard classes
    Store predictions and labels
        ↓
Compute classification metrics
        ↓
If soft labels are available:
    Compute soft cross entropy
    Compute KL similarity
        ↓
Return metrics dictionary
```

### Returned metrics

The evaluator returns:

| Metric | Meaning |
|---|---|
| `loss` | Average validation loss |
| `accuracy` | Overall hard-label correctness |
| `precision` | How many predicted sexist posts were truly sexist |
| `recall` | How many truly sexist posts were detected |
| `f1` | Harmonic mean of precision and recall |
| `confusion_matrix` | 2x2 breakdown of predictions |
| `soft_cross_entropy` | Internal soft-label monitoring metric |
| `kl_similarity` | Internal KL-based similarity score |

### Classification metrics

Classification metrics are implemented in:

```text
src/engine/metrics.py
```

Main function:

```python
compute_classification_metrics(true_labels, predictions)
```

It computes:

1. Accuracy.
2. Precision.
3. Recall.
4. F1-score.
5. Confusion matrix.

The implementation uses:

```python
zero_division=0
```

This avoids warnings or crashes when one class is not predicted.

The confusion matrix uses:

```python
labels=[0, 1]
```

This guarantees a stable 2x2 matrix in the order:

```text
0 = NO
1 = YES
```

### Internal soft metrics

Internal soft metrics are implemented in:

```text
src/engine/soft_metrics.py
```

The file provides:

```python
soft_cross_entropy(true_probs, pred_probs)
kl_similarity_score(true_probs, pred_probs)
```

Important:

```text
These are internal validation-monitoring metrics only.
They are not the official EXIST ICM-Soft metric.
```

The official EXIST ICM and ICM-Soft scores are computed by:

```text
data/evaluation/exist2023evaluation.py
```

through:

```text
evaluate_official.py
```

---

## Prediction and Ensemble Inference

Prediction utilities are implemented in:

```text
src/engine/predictor.py
```

### Predicting probabilities

Function:

```python
predict_probabilities(model, data_loader, device)
```

This function:

1. Sets the model to evaluation mode.
2. Disables gradient calculation.
3. Runs the model on each batch.
4. Applies softmax to logits.
5. Returns probabilities.

Example:

```text
logits:        [-1.2, 2.4]
probabilities: [0.027, 0.973]
```

The output order is:

```text
[NO_probability, YES_probability]
```

### Ensemble averaging

Function:

```python
ensemble_mean(predictions_list)
```

This function averages probability predictions from multiple models.

Example:

```text
Model 1: [0.20, 0.80]
Model 2: [0.40, 0.60]

Ensemble: [0.30, 0.70]
```

This is also known as soft voting.

The project uses this to combine:

```text
xlm-roberta-base
bert-base-multilingual-cased
```

into one final prediction distribution.

---

## Official EXIST Prediction Format

Prediction writing is implemented in:

```text
src/engine/prediction_writer.py
```

Main function:

```python
write_task1_predictions(ids, probabilities, output_path)
```

The function receives:

```text
ids: tweet IDs
probabilities: model or ensemble probabilities in [NO, YES] order
output_path: destination JSON path
```

It writes official EXIST Task 1 JSON predictions.

Example output:

```json
{
    "tweet_id": {
        "id_EXIST": "tweet_id",
        "hard_label": "YES",
        "soft_label": {
            "NO": 0.25,
            "YES": 0.75
        }
    }
}
```

### Hard-label decision rule

The hard label is selected from probabilities:

```python
hard_label = "YES" if yes_prob > no_prob else "NO"
```

This means ties are assigned to:

```text
NO
```

### Defensive probability normalization

Before writing, probabilities are normalized defensively:

```python
total = no_prob + yes_prob
if total > 0:
    no_prob = no_prob / total
    yes_prob = yes_prob / total
```

This protects against tiny numerical issues.

---

## Configuration

Configuration is centralized in:

```text
src/config.py
```

### Models

```python
MODEL_NAMES = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased"
]
```

### Training hyperparameters

```python
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
```

### Training mode

```python
MODE = "soft"
```

Options:

```text
"soft" = train with annotator disagreement probability distributions
"hard" = train with majority-vote hard labels
```

### Task settings

```python
NUM_CLASSES = 2
TASK1_LABELS = ["NO", "YES"]
```

### Device

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

The project automatically uses GPU if CUDA is available.

### Data paths

```python
TRAIN_PATH = "data/training/EXIST2023_training.json"
DEV_PATH = "data/dev/EXIST2023_dev.json"
TEST_PATH = "data/test/EXIST2023_test_clean.json"
```

### Evaluation paths

```python
EVAL_SCRIPT_PATH = "data/evaluation/exist2023evaluation.py"

DEV_TASK1_GOLD_SOFT = "data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json"
DEV_TASK1_GOLD_HARD = "data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json"

TRAIN_TASK1_GOLD_SOFT = "data/evaluation/golds/EXIST2023_training_task1_gold_soft.json"
TRAIN_TASK1_GOLD_HARD = "data/evaluation/golds/EXIST2023_training_task1_gold_hard.json"
```

### Output paths

```python
MODEL_OUTPUT = "outputs/models/"
PREDICTION_OUTPUT = "outputs/predictions/"
```

---

## How to Run the Project

### 1. Install dependencies

Create and activate a Python environment, then install dependencies.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not complete, the core dependencies are:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

### 2. Prepare the data

Make sure the dataset files are placed as configured in `src/config.py`:

```text
data/training/EXIST2023_training.json
data/dev/EXIST2023_dev.json
data/test/EXIST2023_test_clean.json
```

Also make sure the official evaluation files exist:

```text
data/evaluation/exist2023evaluation.py
data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json
data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json
```

### 3. Choose training mode

Open:

```text
src/config.py
```

Set:

```python
MODE = "soft"
```

for learning from disagreement, or:

```python
MODE = "hard"
```

for hard-label baseline training.

The default project mode is:

```python
MODE = "soft"
```

### 4. Train models and create development predictions

Run:

```bash
python train.py
```

This will:

1. Load training and development data.
2. Train each model in `MODEL_NAMES`.
3. Save the best model checkpoints.
4. Generate development-set predictions.
5. Ensemble the model predictions.
6. Write:

```text
outputs/predictions/dev_task1_ensemble_predictions.json
```

### 5. Run official EXIST development evaluation

After training, run:

```bash
python evaluate_official.py
```

This calls:

```text
data/evaluation/exist2023evaluation.py
```

with:

```text
-p outputs/predictions/dev_task1_ensemble_predictions.json
-g data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json
-e data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json
-t task1
```

The official evaluator reports the official Task 1 scores, including disagreement-aware soft evaluation.

### 6. Generate test predictions

After training checkpoints exist, run:

```bash
python predict_test.py
```

This will:

1. Load the test set.
2. Load each trained model checkpoint.
3. Predict probabilities for the test set.
4. Average probabilities across models.
5. Write:

```text
outputs/predictions/test_task1_ensemble_predictions.json
```

The test set has no labels, so this script does not compute metrics.

---

## Outputs

### Model checkpoints

For soft mode:

```text
outputs/models/xlm-roberta-base_soft.pt
outputs/models/bert-base-multilingual-cased_soft.pt
```

For hard mode:

```text
outputs/models/xlm-roberta-base_hard.pt
outputs/models/bert-base-multilingual-cased_hard.pt
```

### Development predictions

```text
outputs/predictions/dev_task1_ensemble_predictions.json
```

### Test predictions

```text
outputs/predictions/test_task1_ensemble_predictions.json
```

### Example prediction file

```json
{
    "12345": {
        "id_EXIST": "12345",
        "hard_label": "YES",
        "soft_label": {
            "NO": 0.23,
            "YES": 0.77
        }
    }
}
```

---

## File-by-File Explanation

### `src/config.py`

Central configuration file.

It defines:

- model names
- hyperparameters
- training mode
- task labels
- device selection
- data paths
- evaluation paths
- output paths

This file should be edited when changing the experiment setup.

---

### `src/data/preprocessing.py`

Handles raw EXIST data loading and label preparation.

Main functions:

```python
compute_soft_label_task1(labels)
load_annotated_data(path)
load_test_data(path)
load_data(path)
```

Responsibilities:

- read JSON files
- compute annotator disagreement distributions
- build `[NO, YES]` label vectors
- create hard labels
- return clean Pandas DataFrames

---

### `src/data/dataset.py`

Defines:

```python
SexismDataset
```

Responsibilities:

- tokenize tweets
- pad/truncate text to `MAX_LEN`
- return `input_ids`
- return `attention_mask`
- return labels when available
- support both soft labels and hard labels
- support test-time inference without labels

---

### `src/models/transformer.py`

Defines:

```python
TransformerModel
```

Responsibilities:

- load a pretrained transformer encoder
- extract token embeddings
- apply mean pooling
- apply max pooling
- concatenate pooled features
- apply dropout
- classify into `NO` or `YES`

The classifier output is a pair of logits:

```text
[NO_score, YES_score]
```

---

### `src/losses/losses.py`

Defines:

```python
SoftLabelLoss
```

This uses:

```python
torch.nn.KLDivLoss(reduction="batchmean")
```

It is used when labels are soft probability distributions.

Hard-label training uses PyTorch's built-in:

```python
torch.nn.CrossEntropyLoss()
```

inside the trainer and evaluator.

---

### `src/engine/trainer.py`

Defines:

```python
train_one_epoch(model, data_loader, optimizer, device)
```

Responsibilities:

- run one training epoch
- compute logits
- select the correct loss function
- backpropagate
- update weights
- track training loss and accuracy

It automatically switches between:

```text
CrossEntropyLoss for hard labels
SoftLabelLoss for soft labels
```

---

### `src/engine/evaluator.py`

Defines:

```python
evaluate(model, data_loader, device)
```

Responsibilities:

- evaluate without weight updates
- compute validation loss
- compute hard classification metrics
- compute internal soft-label metrics when soft labels are available

It uses `model.eval()` and `torch.no_grad()`.

---

### `src/engine/metrics.py`

Defines:

```python
compute_classification_metrics(true_labels, predictions)
```

Returns:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

The metrics are binary metrics where:

```text
0 = NO
1 = YES
```

---

### `src/engine/soft_metrics.py`

Defines internal soft-label monitoring metrics:

```python
soft_cross_entropy(true_probs, pred_probs)
kl_similarity_score(true_probs, pred_probs)
```

Important:

```text
These are not the official EXIST ICM-Soft metric.
```

They are useful for development-time monitoring only.

---

### `src/engine/predictor.py`

Defines:

```python
predict_probabilities(model, data_loader, device)
ensemble_mean(predictions_list)
```

Responsibilities:

- run inference
- convert logits to probabilities
- average probabilities across multiple models

---

### `src/engine/prediction_writer.py`

Defines:

```python
write_task1_predictions(ids, probabilities, output_path)
```

Responsibilities:

- convert probabilities into official EXIST Task 1 format
- create both hard and soft labels
- save JSON prediction files

---

### `train.py`

Main training and development prediction script.

Responsibilities:

- load train and dev data
- choose soft or hard labels based on `MODE`
- train all models in `MODEL_NAMES`
- evaluate after each epoch
- save best checkpoints
- create dev predictions
- ensemble dev predictions
- save official dev prediction JSON
- print ensemble classification metrics

Run with:

```bash
python train.py
```

---

### `evaluate_official.py`

Runs the official EXIST evaluator.

Responsibilities:

- load the configured prediction path
- call the official evaluation script using `subprocess`
- print official evaluation output and warnings/errors

Run with:

```bash
python evaluate_official.py
```

---

### `predict_test.py`

Generates final test predictions.

Responsibilities:

- load test data
- load trained model checkpoints
- predict test probabilities
- ensemble model outputs
- write official test prediction JSON

Run with:

```bash
python predict_test.py
```

---

## Implementation Notes

### Why multilingual transformers?

The EXIST dataset contains both English and Spanish text. The selected models are multilingual and can process both languages:

```text
xlm-roberta-base
bert-base-multilingual-cased
```

This avoids building separate language-specific pipelines.

### Why mean pooling and max pooling?

The model does not use only the first token representation. Instead, it uses two pooling methods:

```text
Mean pooling: captures the overall meaning of the tweet.
Max pooling: captures the strongest or most salient feature.
```

Concatenating them gives the classifier a richer representation:

```text
[mean_pool, max_pool] → classifier
```

### Why KL Divergence for soft labels?

Soft labels are probability distributions. KL Divergence measures the difference between probability distributions, so it is suitable for learning from annotator disagreement.

### Why ensemble averaging?

Different transformer models may make different errors. Averaging their probability outputs can produce more stable predictions than using a single model.

### Important distinction: internal soft metrics vs official EXIST metrics

The project includes:

```text
soft_cross_entropy
kl_similarity_score
```

These are useful during training and validation, but they are not the official EXIST ICM-Soft score.

Official scoring should be performed with:

```bash
python evaluate_official.py
```

which calls the official EXIST evaluation script.

### Known implementation note for hard mode

The current configuration uses:

```python
MODE = "soft"
```

In the provided `src/engine/evaluator.py`, the average validation loss and `kl_similarity` are assigned inside the soft-label branch. If `MODE = "hard"` is used, the evaluator should be checked to ensure `avg_loss` and `kl_similarity` are initialized and returned correctly for hard-label evaluation.

A safer evaluator structure would initialize:

```python
avg_loss = total_loss / len(data_loader)
kl_similarity = None
```

outside the soft-label-only branch.

This does not affect the default `MODE = "soft"` setup, but it is important if running the hard-label baseline.

---

## Current Scope

The current implementation focuses on:

```text
Task 1: Sexism Identification
```

This is a binary classification task:

```text
NO  = non-sexist
YES = sexist
```

The project does not currently implement:

```text
Task 2: Source Intention
Task 3: Sexism Categorization
```

However, the code structure is modular and can be extended later.

---

## Possible Future Improvements

1. Add Task 2 support:
   - possible labels: `NO`, `DIRECT`, `REPORTED`, `JUDGEMENTAL`
   - update `NUM_CLASSES`
   - update preprocessing and prediction writer

2. Add Task 3 support:
   - support sexism category classification
   - adapt output layer and label handling
   - possibly use multi-label classification if required

3. Train for more epochs:
   - current default is `EPOCHS = 2`
   - longer training may improve performance

4. Tune hyperparameters:
   - learning rate
   - batch size
   - dropout
   - maximum sequence length
   - optimizer settings

5. Add language-specific analysis:
   - evaluate English posts separately
   - evaluate Spanish posts separately

6. Add model ablation studies:
   - XLM-RoBERTa only
   - mBERT only
   - ensemble

7. Compare hard-label and soft-label training:
   - run once with `MODE = "hard"`
   - run once with `MODE = "soft"`
   - compare official EXIST scores

8. Add experiment logging:
   - save training curves
   - save validation metrics per epoch
   - use CSV, TensorBoard, or Weights & Biases

9. Add reproducibility controls:
   - random seeds
   - deterministic PyTorch settings
   - environment/version documentation

10. Improve error handling:
   - validate file paths before training
   - check prediction shape before writing
   - handle missing checkpoints more gracefully

---

## Summary

This project is a complete EXIST 2023 Task 1 pipeline for multilingual sexism detection under the learning-from-disagreement paradigm.

It:

```text
loads official EXIST data
preprocesses annotator disagreement
creates soft and hard labels
trains multilingual transformer models
supports soft-label and hard-label training
uses KL Divergence for soft-label learning
evaluates with standard classification metrics
ensembles model predictions
writes official EXIST Task 1 JSON files
runs the official EXIST evaluation script
generates final test predictions
```

The main contribution is that the model does not ignore disagreement between annotators. Instead, it learns from the full annotator distribution, making the system better aligned with subjective and ambiguous classification tasks such as sexism detection.
