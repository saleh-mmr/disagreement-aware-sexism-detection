# src/engine/evaluator.py

import torch
from tqdm import tqdm
from src.engine.metrics import compute_classification_metrics
from src.losses.losses import SoftLabelLoss
from src.engine.soft_metrics import soft_cross_entropy, icm_soft
import numpy as np

'''
This file evaluates the model without updating weights.
'''

def evaluate(model, data_loader, device):
    # This disables dropout and puts the model in evaluation mode.
    model.eval()

    total_loss = 0.0

    all_preds = []
    all_true = []
    all_probs = []
    all_true_soft = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    # This prevents gradient calculation, which makes evaluation faster and saves memory.
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # For each batch, it gets logits.
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

            if labels.dtype == torch.long:
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(logits, labels)
                true = labels.cpu().numpy()
            else:
                criterion = SoftLabelLoss()
                loss = criterion(logits, labels)

                true = torch.argmax(labels, dim=1).cpu().numpy()

                # NEW: store soft labels
                all_true_soft.extend(labels.cpu().numpy())

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(true)


    # Important limitation
    # The proposal says we should also evaluate soft-label behavior using soft metrics such as cross-entropy or ICM-Soft.
    # ICM-Soft should be implemented

    metrics = compute_classification_metrics(all_true, all_preds)
    soft_ce = None
    icm = None

    if len(all_true_soft) > 0:
        true_soft_array = np.array(all_true_soft)
        probs_array = np.array(all_probs)

        soft_ce = soft_cross_entropy(true_soft_array, probs_array)
        icm = icm_soft(true_soft_array, probs_array)

        avg_loss = total_loss / len(data_loader)

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "soft_cross_entropy": soft_ce,
        "icm_soft": icm
    }