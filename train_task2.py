import os
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import (
    MODEL_NAMES,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LR,
    DEVICE,
    TRAIN_PATH,
    DEV_PATH,
    TASK2_NUM_CLASSES,
    TASK2_LABELS,
    TASK2_MODEL_OUTPUT,
    TASK2_PREDICTION_OUTPUT
)
from src.data.preprocessing import load_annotated_data_task2
from src.data.dataset import SexismDataset
from src.models.transformer import TransformerModel
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate
from src.engine.predictor import predict_probabilities, ensemble_mean
from src.engine.task2_prediction_writer import write_task2_predictions


def compute_multiclass_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)

    precision_macro = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(TASK2_LABELS)))
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm
    }


def evaluate_task2(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_true = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_multiclass_metrics(all_true, all_preds)

    return {
        "loss": avg_loss,
        **metrics
    }


def train_single_task2_model(model_name, train_loader, dev_loader):
    print(f"\n===== Training Task 2 model: {model_name} =====")

    model = TransformerModel(model_name, TASK2_NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE
        )

        val_metrics = evaluate_task2(
            model,
            dev_loader,
            DEVICE
        )

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision Macro: {val_metrics['precision_macro']:.4f}")
        print(f"Val Recall Macro: {val_metrics['recall_macro']:.4f}")
        print(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
        print(f"Val F1 Weighted: {val_metrics['f1_weighted']:.4f}")
        print("Val Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            path = TASK2_MODEL_OUTPUT + f"{model_name.replace('/', '_')}_task2.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved best Task 2 model: {path}")

    return model


def main():
    os.makedirs(TASK2_MODEL_OUTPUT, exist_ok=True)
    os.makedirs(TASK2_PREDICTION_OUTPUT, exist_ok=True)

    print("Loading Task 2 training data...")
    train_df = load_annotated_data_task2(TRAIN_PATH)

    print("Loading Task 2 development data...")
    dev_df = load_annotated_data_task2(DEV_PATH)

    train_texts = train_df["text"].values
    dev_texts = dev_df["text"].values

    train_labels = train_df["task2_hard_label"].values
    dev_labels = dev_df["task2_hard_label"].values

    all_predictions = []

    for model_name in MODEL_NAMES:
        print("\n==============================")
        print(f"Processing Task 2 model: {model_name}")
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = SexismDataset(
            train_texts,
            train_labels,
            tokenizer,
            MAX_LEN
        )

        dev_dataset = SexismDataset(
            dev_texts,
            dev_labels,
            tokenizer,
            MAX_LEN
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = train_single_task2_model(
            model_name,
            train_loader,
            dev_loader
        )

        preds = predict_probabilities(
            model,
            dev_loader,
            DEVICE
        )

        all_predictions.append(preds)

    print("\n===== Task 2 Ensemble Results on Dev Set =====")

    ensemble_preds = ensemble_mean(all_predictions)
    ensemble_pred_labels = np.argmax(ensemble_preds, axis=1)

    metrics = compute_multiclass_metrics(
        dev_labels,
        ensemble_pred_labels
    )

    print(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
    print(f"Ensemble Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Ensemble Recall Macro: {metrics['recall_macro']:.4f}")
    print(f"Ensemble F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"Ensemble F1 Weighted: {metrics['f1_weighted']:.4f}")
    print("Ensemble Confusion Matrix:")
    print(metrics["confusion_matrix"])

    prediction_path = TASK2_PREDICTION_OUTPUT + "dev_task2_ensemble_predictions.json"

    write_task2_predictions(
        ids=dev_df["id_EXIST"].values,
        probabilities=ensemble_preds,
        output_path=prediction_path
    )


if __name__ == "__main__":
    main()