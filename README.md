# Project Explanation: Learning from Disagreement for Multilingual Sexism Detection

## 1. Project Overview

This project implements a multilingual sexism detection system based on the **EXIST 2023** dataset. The main idea is to detect sexist content in English and Spanish social media posts while also modeling **annotator disagreement**.

Traditional classification systems usually convert multiple human annotations into one single hard label, such as:

```text
YES = sexist
NO = non-sexist
```

However, sexism detection is subjective. Different annotators may disagree about whether a tweet is sexist. Therefore, this project follows the **learning from disagreement** paradigm:

```text
Instead of training only on one final label,
the model learns from the distribution of annotator opinions.
```

Example:

```text
Annotator labels: ["YES", "YES", "NO"]

Soft label:
NO  = 1 / 3 = 0.3333
YES = 2 / 3 = 0.6667
```

So instead of training the model with only:

```text
YES
```

the model learns:

```text
{"NO": 0.3333, "YES": 0.6667}
```

This allows the model to better represent uncertainty and human disagreement.

---

## 2. Final Project Goal

The final system:

1. Loads the official EXIST training, development, and test datasets.
2. Converts annotator labels into soft probability distributions.
3. Trains multilingual transformer models:
   - `xlm-roberta-base`
   - `bert-base-multilingual-cased`
4. Supports both:
   - hard-label training
   - soft-label training
5. Uses KL Divergence loss for soft-label learning.
6. Combines model predictions using ensemble averaging.
7. Saves predictions in the official EXIST Task 1 JSON format.
8. Runs the official EXIST evaluator to calculate official ICM / ICM-Soft metrics.
9. Generates predictions for the test set.

---

## 3. Final Project Structure

```text
akbarhalvaeirezaei2526/
├── explanation.md
├── requirements.txt
├── train.py
├── evaluate_official.py
├── predict_test.py
├── data/
│   ├── training/
│   │   └── EXIST2023_training.json
│   ├── dev/
│   │   └── EXIST2023_dev.json
│   ├── test/
│   │   └── EXIST2023_test_clean.json
│   └── evaluation/
│       ├── exist2023evaluation.py
│       ├── baselines/
│       └── golds/
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
    ├── models/
    │   └── transformer.py
    └── utils/
        └── utils.py
```

---

## 4. Dataset Structure and Usage

The project uses three main dataset splits:

```text
data/training/EXIST2023_training.json
data/dev/EXIST2023_dev.json
data/test/EXIST2023_test_clean.json
```

### 4.1 Training Set

Used to train the models.

```text
data/training/EXIST2023_training.json
```

This file contains tweets and annotator labels.

### 4.2 Development Set

Used for validation and official evaluation.

```text
data/dev/EXIST2023_dev.json
```

The model generates predictions for this file. These predictions are compared against the official gold files:

```text
data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json
data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json
```

### 4.3 Test Set

Used only for generating final predictions.

```text
data/test/EXIST2023_test_clean.json
```

The test set does not include labels, so the project cannot compute supervised metrics on it. It only generates a prediction file:

```text
outputs/predictions/test_task1_ensemble_predictions.json
```

---

## 5. Development Process Step by Step

## Step 1: Data Preprocessing

The first step was to read the raw EXIST JSON files and convert annotator labels into usable training labels.

File:

```text
src/data/preprocessing.py
```

This file creates:

1. soft labels
2. label vectors
3. hard labels
4. metadata such as ID, language, and split

Example input:

```text
labels_task1 = ["YES", "NO", "YES"]
```

Output:

```python
soft_label = {
    "NO": 0.3333,
    "YES": 0.6667
}

label_vector = [0.3333, 0.6667]

hard_label = 1
```

The project uses this label order:

```text
index 0 = NO
index 1 = YES
```

This order is used consistently in training, prediction, and JSON output writing.

---

## Step 2: Dataset Class

File:

```text
src/data/dataset.py
```

This file defines the PyTorch dataset class:

```python
SexismDataset
```

Its job is to:

1. receive texts and labels
2. tokenize texts using a HuggingFace tokenizer
3. return PyTorch tensors

Each item contains:

```python
{
    "input_ids": tensor,
    "attention_mask": tensor,
    "labels": tensor
}
```

For test prediction, labels are optional. This means the same dataset class can be used for:

```text
training/dev data with labels
test data without labels
```

The dataset automatically chooses label type:

```text
hard label → torch.long
soft label → torch.float
```

This is important because the trainer uses the label type to decide which loss function to use.

---

## Step 3: Transformer Model

File:

```text
src/models/transformer.py
```

This file defines the model architecture:

```python
TransformerModel
```

The model uses a pretrained multilingual transformer encoder:

```python
AutoModel.from_pretrained(model_name)
```

The project trains two multilingual models:

```text
xlm-roberta-base
bert-base-multilingual-cased
```

This is appropriate because EXIST contains both English and Spanish tweets.

### Model Flow

```text
input_ids + attention_mask
        ↓
pretrained transformer encoder
        ↓
last hidden states
        ↓
mean pooling + max pooling
        ↓
concatenation
        ↓
dropout
        ↓
linear classifier
        ↓
logits
```

### Mean Pooling

Mean pooling averages token embeddings while ignoring padding tokens.

It captures the general meaning of the sentence.

### Max Pooling

Max pooling extracts the strongest feature value across tokens.

It captures the most important signal in the tweet.

### Max Pooling Bug Fix

Originally, max pooling modified the transformer hidden states in-place:

```python
hidden_states[mask == 0] = -1e9
```

This is risky because PyTorch autograd may need the original tensor values during backpropagation.

The fixed version uses:

```python
masked_hidden_states = hidden_states.masked_fill(~mask, -1e9)
```

This creates a masked copy instead of modifying the original tensor.

---

## Step 4: Loss Function

File:

```text
src/losses/losses.py
```

This file defines:

```python
SoftLabelLoss
```

For hard labels, the project uses:

```python
CrossEntropyLoss
```

For soft labels, the project uses:

```python
KLDivLoss
```

Soft-label training compares two probability distributions:

```text
target distribution:   human annotator disagreement
predicted distribution: model output probabilities
```

Example:

```text
target = [0.3, 0.7]
prediction = [0.2, 0.8]
```

The model is not only learning which class is correct. It is learning how strongly the tweet belongs to each class.

---

## Step 5: Training Logic

File:

```text
src/engine/trainer.py
```

This file trains the model for one epoch.

The trainer does the following:

1. sets model to training mode
2. loops through batches
3. sends tensors to CPU/GPU
4. computes logits
5. chooses the correct loss function
6. performs backpropagation
7. updates model weights
8. tracks loss and accuracy

The important design is automatic loss selection:

```python
if labels.dtype == torch.long:
    use CrossEntropyLoss
else:
    use SoftLabelLoss
```

So the same training loop works for both hard and soft training.

---

## Step 6: Evaluation Logic

File:

```text
src/engine/evaluator.py
```

This file evaluates a trained model without updating weights.

It uses:

```python
model.eval()
torch.no_grad()
```

This disables dropout and prevents gradient calculation.

The evaluator returns:

```text
loss
accuracy
precision
recall
f1
confusion matrix
soft cross entropy
KL similarity
```

For soft-label evaluation, the loss is computed using the soft labels, but classification metrics are computed by converting the soft label into a hard label using:

```python
argmax(label_vector)
```

Example:

```text
[0.3, 0.7] → YES
[0.8, 0.2] → NO
```

---

## Step 7: Classification Metrics

File:

```text
src/engine/metrics.py
```

This file computes standard classification metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. Confusion matrix

The metrics use:

```python
zero_division=0
```

This prevents warnings or crashes when the model predicts only one class.

The confusion matrix is forced to always be 2x2:

```python
labels=[0, 1]
```

This makes the output stable.

---

## Step 8: Internal Soft Metrics

File:

```text
src/engine/soft_metrics.py
```

This file provides internal monitoring metrics only.

It includes:

```python
soft_cross_entropy
kl_similarity_score
```

Important distinction:

```text
soft_cross_entropy and KL similarity are internal monitoring metrics.
They are not the official EXIST ICM-Soft metric.
```

The official ICM-Soft is computed using:

```text
data/evaluation/exist2023evaluation.py
```

This distinction is important because the official EXIST metric is based on information content and hierarchy-aware ICM, not a simple KL similarity.

---

## Step 9: Prediction Utilities

File:

```text
src/engine/predictor.py
```

This file has two main functions.

### predict_probabilities

This function:

1. runs the model in evaluation mode
2. computes logits
3. applies softmax
4. returns probabilities

Example:

```text
logits = [-1.2, 2.4]
probabilities = [0.027, 0.973]
```

### ensemble_mean

This function averages predictions from multiple models.

Example:

```text
Model 1: [0.20, 0.80]
Model 2: [0.40, 0.60]

Ensemble: [0.30, 0.70]
```

This is called soft voting.

---

## Step 10: Official Prediction Writer

File:

```text
src/engine/prediction_writer.py
```

This file converts model probabilities into the official EXIST Task 1 JSON format.

The model outputs:

```text
[NO_probability, YES_probability]
```

The official format needs:

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

This file creates exactly that format.

The hard label is selected as:

```python
"YES" if yes_prob > no_prob else "NO"
```

---

## Step 11: Configuration

File:

```text
src/config.py
```

This file centralizes important project settings.

It contains:

```python
MODEL_NAMES = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased"
]
```

Training hyperparameters:

```python
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
```

Task settings:

```python
NUM_CLASSES = 2
TASK1_LABELS = ["NO", "YES"]
MODE = "soft"
```

Paths:

```python
TRAIN_PATH = "data/training/EXIST2023_training.json"
DEV_PATH = "data/dev/EXIST2023_dev.json"
TEST_PATH = "data/test/EXIST2023_test_clean.json"
```

Official evaluation paths:

```python
EVAL_SCRIPT_PATH = "data/evaluation/exist2023evaluation.py"
DEV_TASK1_GOLD_SOFT = "data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json"
DEV_TASK1_GOLD_HARD = "data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json"
```

Output paths:

```python
MODEL_OUTPUT = "outputs/models/"
PREDICTION_OUTPUT = "outputs/predictions/"
```

The `MODE` variable controls whether the project trains using hard labels or soft labels.

---

## Step 12: Main Training Pipeline

File:

```text
train.py
```

This is the main script.

Run it with:

```bash
python train.py
```

It performs the full training and dev prediction process.

### train.py flow

```text
create output folders
        ↓
load training data
        ↓
load development data
        ↓
choose hard or soft labels
        ↓
for each model:
    create tokenizer
    create dataset
    create dataloader
    train model
    evaluate model
    save best checkpoint
    generate dev probabilities
        ↓
ensemble model probabilities
        ↓
save official dev prediction JSON
        ↓
print ensemble classification metrics
```

### Saved model files

After training, the project saves:

```text
outputs/models/xlm-roberta-base_soft.pt
outputs/models/bert-base-multilingual-cased_soft.pt
```

If `MODE = "hard"`, the filenames become:

```text
outputs/models/xlm-roberta-base_hard.pt
outputs/models/bert-base-multilingual-cased_hard.pt
```

---

## Step 13: Official Evaluation

File:

```text
evaluate_official.py
```

Run it with:

```bash
python evaluate_official.py
```

This script calls:

```text
data/evaluation/exist2023evaluation.py
```

with the correct arguments:

```text
-p outputs/predictions/dev_task1_ensemble_predictions.json
-g data/evaluation/golds/EXIST2023_dev_task1_gold_soft.json
-e data/evaluation/golds/EXIST2023_dev_task1_gold_hard.json
-t task1
```

The official evaluator reports:

1. ICM hard-hard
2. FMeasure hard-hard
3. ICM hard-soft
4. ICM soft-soft

This is the official way to evaluate learning from disagreement for EXIST.

---

## Step 14: Test Prediction

File:

```text
predict_test.py
```

Run it with:

```bash
python predict_test.py
```

This script:

1. loads the test file
2. loads the trained model checkpoints
3. predicts probabilities
4. averages predictions using ensemble mean
5. writes the official Task 1 JSON output

Output:

```text
outputs/predictions/test_task1_ensemble_predictions.json
```

The test set has no labels, so no metrics are computed.

---

## Step 15: Utility File

File:

```text
src/utils/utils.py
```

This file contains:

```python
filter_by_no_value
```

It filters out samples where the `NO` probability is above a threshold.

Example:

```python
filter_by_no_value(texts, labels, no_values, threshold=0.8)
```

This can be used to remove examples that are very confidently non-sexist and focus training on more ambiguous examples.

Currently, this function is available but not part of the main training pipeline.

---

## 6. Final Commands

Run commands in this order:

```bash
python train.py
```

Then:

```bash
python evaluate_official.py
```

Then:

```bash
python predict_test.py
```

---

## 7. Final Results

The final training run produced these development-set ensemble results:

```text
Ensemble Accuracy:  0.7958
Ensemble Precision: 0.7591
Ensemble Recall:    0.7824
Ensemble F1:        0.7706
```

Official EXIST evaluation results:

```text
ICM hard-hard:   0.4807
F1 YES:          0.7706
F1 NO:           0.7958
Macro-F:         0.7832
ICM hard-soft:   0.1506
ICM soft-soft:   0.5473
```

The most important disagreement-aware result is:

```text
ICM soft-soft: 0.5473
```

This means the model is evaluated not only by whether it predicts the majority label, but also by how close its predicted probability distribution is to the human annotator distribution.

---

## 8. How This Matches the Proposal

The proposal described a system for:

```text
Learning from Disagreement: A Multilingual Ensemble Approach for Sexism Detection in Social Media
```

The final implementation matches the proposal as follows:

| Proposal Goal | Implementation |
|---|---|
| Dataset understanding and preprocessing | `src/data/preprocessing.py` |
| Soft label distributions | `label_vector = [NO, YES]` |
| Hard-label baseline support | `MODE = "hard"` |
| Soft-label learning | `MODE = "soft"` + `SoftLabelLoss` |
| Transformer models | XLM-RoBERTa + mBERT |
| Multilingual support | English and Spanish transformer models |
| Ensemble model | `ensemble_mean()` |
| Official Task 1 prediction format | `prediction_writer.py` |
| Official ICM / ICM-Soft evaluation | `evaluate_official.py` + `exist2023evaluation.py` |
| Test prediction generation | `predict_test.py` |

---

## 9. Current Project Scope

The current implementation focuses on:

```text
Task 1: Sexism Identification
```

This is a binary classification task:

```text
NO  = non-sexist
YES = sexist
```

The project does not yet implement:

```text
Task 2: Source Intention
Task 3: Sexism Categorization
```

However, the current code structure is modular and can be extended to support those tasks later.

---

## 10. Possible Future Improvements

1. Add Task 2 support:
   - labels: `NO`, `DIRECT`, `REPORTED`, `JUDGEMENTAL`
   - `NUM_CLASSES = 4`

2. Add Task 3 support:
   - labels: `NO`, `IDEOLOGICAL-INEQUALITY`, `STEREOTYPING-DOMINANCE`, `OBJECTIFICATION`, `SEXUAL-VIOLENCE`, `MISOGYNY-NON-SEXUAL-VIOLENCE`
   - multi-label output
   - sigmoid instead of softmax

3. Train for more epochs.

4. Tune hyperparameters:
   - learning rate
   - batch size
   - dropout
   - max sequence length

5. Add language-specific analysis:
   - English results
   - Spanish results

6. Add ablation study:
   - XLM-R only
   - mBERT only
   - ensemble

7. Add hard-vs-soft comparison:
   - train once with `MODE = "hard"`
   - train once with `MODE = "soft"`
   - compare official scores

---

## 11. Summary

This project is a complete Task 1 EXIST 2023 pipeline for multilingual sexism detection under the learning from disagreement paradigm.

It:

```text
loads official data
preprocesses annotator disagreement
trains multilingual transformer models
supports hard and soft labels
uses KL divergence for soft-label learning
ensembles predictions
writes official prediction JSON files
runs official EXIST evaluation
generates test predictions
```

The project is now aligned with the proposal and produces official disagreement-aware evaluation results.
