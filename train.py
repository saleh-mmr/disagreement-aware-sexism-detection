# train.py

import os
import torch
from transformers import AutoTokenizer
from src.engine.metrics import compute_classification_metrics
import numpy as np
from src.config import *  # Import hyperparameters and paths (LR, DEVICE, EPOCHS, etc.)
from src.data.preprocessing import load_annotated_data
from src.data.dataset import SexismDataset
from src.models.transformer import TransformerModel
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate
from src.engine.predictor import predict_probabilities, ensemble_mean
from src.engine.prediction_writer import write_task1_predictions

def train_single_model(model_name, train_loader, val_loader):
    """
    Handles the training loop for an individual Transformer model.
    """
    print(f"\n===== Training model: {model_name} =====")

    # Initialize tokenizer and model architecture
    model = TransformerModel(model_name, NUM_CLASSES)
    model.to(DEVICE)

    # AdamW is the standard optimizer for Transformer architectures
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Run training step for one pass over the dataset
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, DEVICE
        )

        # Run validation step
        val_metrics = evaluate(
            model, val_loader, DEVICE
        )

        # Log performance statistics
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        # Soft-label evaluation metrics
        if MODE == "soft":
            print(f"Soft Cross Entropy: {val_metrics['soft_cross_entropy']:.4f}")
            print(f"KL Similarity: {val_metrics['kl_similarity']:.4f}")
        print("Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        # Checkpoint: Save model if validation loss improves
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            # Sanitize model name for file path
            path = MODEL_OUTPUT + f"{model_name.replace('/', '_')}_{MODE}.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved best model: {path}")

    return model

def main():
    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    os.makedirs(PREDICTION_OUTPUT, exist_ok=True)
    
    print("Loading training data...")
    train_df = load_annotated_data(TRAIN_PATH)

    print("Loading development data...")
    dev_df = load_annotated_data(DEV_PATH)

    # LABEL SELECTION: Use probability vectors (soft) or discrete integers (hard)
    if MODE == "soft":
        train_labels = train_df["label_vector"].values
        dev_labels = dev_df["label_vector"].values
    else:
        train_labels = train_df["hard_label"].values
        dev_labels = dev_df["hard_label"].values

    train_texts = train_df["text"].values
    dev_texts = dev_df["text"].values

    all_models = []
    all_predictions = []

    # TRAIN MULTIPLE MODELS: Iterate through model list defined in src.config
    for model_name in MODEL_NAMES:
        print("\n==============================")
        print(f"Processing model: {model_name}")
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare PyTorch Datasets
        train_dataset = SexismDataset(
            train_texts, train_labels, tokenizer, MAX_LEN
        )

        dev_dataset = SexismDataset(
            dev_texts, dev_labels, tokenizer, MAX_LEN
        )

        # Create DataLoaders for batch processing
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

        # Execute training process
        model = train_single_model(model_name, train_loader, dev_loader)
        all_models.append(model)

        # Generate probabilities on the official dev set for later ensembling
        preds = predict_probabilities(model, dev_loader, DEVICE)
        all_predictions.append(preds)

    # ENSEMBLE: Combine predictions from all trained models
    print("\n===== ENSEMBLE RESULTS ON DEV SET =====")

    # Calculate the mean of probabilities across all models
    ensemble_preds = ensemble_mean(all_predictions)
    # after running we should get outputs/predictions/dev_task1_ensemble_predictions.json
    # this file contains:
    # "tweet_id": {
    #     "id_EXIST": "tweet_id",
    #     "hard_label": "YES",
    #     "soft_label": {
    #         "NO": 0.23,
    #         "YES": 0.77
    # }}
    dev_prediction_path = PREDICTION_OUTPUT + "dev_task1_ensemble_predictions.json"
    write_task1_predictions(
        ids=dev_df["id_EXIST"].values,
        probabilities=ensemble_preds,
        output_path=dev_prediction_path
    )

    print("\n===== ENSEMBLE EVALUATION =====")

    # Convert probability distributions back to class labels
    ensemble_preds_labels = np.argmax(ensemble_preds, axis=1)

    # Determine ground truth for evaluation
    if MODE == "soft":
        true_labels = np.array([int(x[1] > x[0]) for x in dev_labels])
    else:
        true_labels = dev_labels

    # Calculate and display final ensemble performance
    metrics = compute_classification_metrics(true_labels, ensemble_preds_labels)

    print(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
    print(f"Ensemble Precision: {metrics['precision']:.4f}")
    print(f"Ensemble Recall: {metrics['recall']:.4f}")
    print(f"Ensemble F1: {metrics['f1']:.4f}")

    print("Ensemble Confusion Matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()