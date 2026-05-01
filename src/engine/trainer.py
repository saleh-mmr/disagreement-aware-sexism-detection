import torch
from tqdm import tqdm

"""
Logic for training the model for a single epoch. 
It automatically switches between standard CrossEntropy for hard labels 
and a custom SoftLabelLoss for learning from annotator disagreement.
"""

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # AUTO LOSS SELECTION
        if labels.dtype == torch.long:
            # HARD LABELS → BASELINE
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()

        else:
            # SOFT LABELS → DISAGREEMENT LEARNING
            from src.losses.losses import SoftLabelLoss
            criterion = SoftLabelLoss()

            loss = criterion(logits, labels)

            # convert soft → hard for monitoring
            hard_labels = torch.argmax(labels, dim=1)
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == hard_labels).sum().item()

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