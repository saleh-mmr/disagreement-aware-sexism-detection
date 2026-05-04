# src/engine/trainer.py

import torch
from tqdm import tqdm
from src.losses.losses import SoftLabelLoss

"""
Logic for training the model for a single epoch. 
It automatically switches between standard CrossEntropy for hard labels 
and a custom SoftLabelLoss for learning from annotator disagreement.
"""

def train_one_epoch(model, data_loader, optimizer, device):
    # Set model to training mode and enables dropout and gradient updates.
    model.train()

    # Initializes tracking variables
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass through the model to get logits
        # Logits are raw class scores, not probabilities yet
        # The model outputs a tensor (a set of numbers) for every tweet in the batch.
        # For each tweet, we get two numbers:
        # Logit 0: The raw score for "non-sexist".
        # Logit 1: The raw score for "sexist".
        # These numbers can be anything (e.g., -2.45, 1.20, 5.8).
        # They are not percentages yet; they are just "strength indicators".
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # AUTO LOSS SELECTION
        if labels.dtype == torch.long:
            # HARD LABELS → BASELINE
            # CrossEntropyLoss expects integer class labels, not one-hot or soft labels
            # It combines LogSoftmax and NLLLoss in one single class, so we can feed raw logits directly.
            # The loss will be high if the predicted class (highest logit) does not match the true class label.
            # For monitoring accuracy, we convert logits to predicted class indices and compare with true labels.
            # This is the standard approach for classification tasks with hard labels.
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()
        else:
            # SOFT LABELS → DISAGREEMENT LEARNING
            criterion = SoftLabelLoss()
            loss = criterion(logits, labels)
            # convert soft → hard for monitoring
            # For accuracy calculation, we need to convert the soft labels (which are distributions) into
            # hard labels (the class with the highest probability).
            hard_labels = torch.argmax(labels, dim=1)
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == hard_labels).sum().item()

        # Backpropagation which updates the model weights.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{(total_correct / total_samples):.4f}"
        )

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }