# src/engine/evaluator.py

import torch
from tqdm import tqdm
from src.engine.metrics import compute_classification_metrics


def evaluate(model, data_loader, device):
    model.eval()

    total_loss = 0.0

    all_preds = []
    all_true = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # LOSS
            if labels.dtype == torch.long:
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(logits, labels)

                true = labels.cpu().numpy()

            else:
                from src.losses.losses import SoftLabelLoss
                criterion = SoftLabelLoss()
                loss = criterion(logits, labels)

                true = torch.argmax(labels, dim=1).cpu().numpy()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(true)

    metrics = compute_classification_metrics(all_true, all_preds)

    avg_loss = total_loss / len(data_loader)

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"]
    }