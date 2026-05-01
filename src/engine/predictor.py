import torch
import numpy as np


def predict_probabilities(model, data_loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return all_probs


def ensemble_mean(predictions_list):
    """
    Average predictions from multiple models.

    predictions_list example:
    [
        preds_from_model_1,
        preds_from_model_2
    ]
    """

    predictions_array = np.array(predictions_list)
    mean_predictions = predictions_array.mean(axis=0)

    return mean_predictions