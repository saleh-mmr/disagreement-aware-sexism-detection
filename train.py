# train.py

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.engine.metrics import compute_classification_metrics
import numpy as np
from src.config import *
from src.data.preprocessing import load_data
from src.data.dataset import SexismDataset
from src.models.transformer import TransformerModel
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate
from src.engine.predictor import predict_probabilities, ensemble_mean

# SWITCH MODE
MODE = "hard"  # "hard" or "soft"


def train_single_model(model_name, train_loader, val_loader):
    print(f"\n===== Training model: {model_name} =====")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TransformerModel(model_name, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, DEVICE
        )

        val_metrics = evaluate(
            model, val_loader, DEVICE
        )

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")

        print("Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            path = MODEL_OUTPUT + f"{model_name.replace('/', '_')}_{MODE}.pt"
            torch.save(model.state_dict(), path)

            print(f"Saved best model: {path}")

    return model


def main():
    print("Loading data...")
    df = load_data(TRAIN_PATH)

    # LABEL SELECTION
    if MODE == "soft":
        labels = df["label_vector"].values
    else:
        labels = df["hard_label"].values

    texts = df["text"].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=df["hard_label"].values
    )

    all_models = []
    all_predictions = []

    # TRAIN MULTIPLE MODELS
    for model_name in MODEL_NAMES:
        print("\n==============================")
        print(f"Processing model: {model_name}")
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = SexismDataset(
            train_texts, train_labels, tokenizer, MAX_LEN
        )

        val_dataset = SexismDataset(
            val_texts, val_labels, tokenizer, MAX_LEN
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE
        )

        model = train_single_model(model_name, train_loader, val_loader)

        all_models.append(model)

        # PREDICT ON VALIDATION SET
        preds = predict_probabilities(model, val_loader, DEVICE)
        all_predictions.append(preds)

    # ENSEMBLE
    print("\n===== ENSEMBLE RESULTS =====")
    
    ensemble_preds = ensemble_mean(all_predictions)

    print("\n===== ENSEMBLE EVALUATION =====")

    # convert probabilities → predicted labels
    ensemble_preds_labels = np.argmax(ensemble_preds, axis=1)

    # true labels (always hard labels for evaluation)
    true_labels = val_labels
    if MODE == "soft":
        true_labels = np.array([int(x[1] > x[0]) for x in val_labels])

    metrics = compute_classification_metrics(true_labels, ensemble_preds_labels)

    print(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
    print(f"Ensemble Precision: {metrics['precision']:.4f}")
    print(f"Ensemble Recall: {metrics['recall']:.4f}")
    print(f"Ensemble F1: {metrics['f1']:.4f}")

    print("Ensemble Confusion Matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()