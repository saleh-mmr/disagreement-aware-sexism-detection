# src/engine/predictor.py


import torch
import numpy as np

# This function gets model predictions as probabilities
def predict_probabilities(model, data_loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # model predictions (logits) e.g. [-2.5, 4.7]
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # probabilities e.g. [0.3, 0.7]
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return all_probs


def ensemble_mean(predictions_list):
    """
    Average predictions from multiple models.

    predictions_list example:
        model 1 prediction = [0.20, 0.80]
        model 2 prediction = [0.40, 0.60]

        ensemble = [0.30, 0.70]
    """

    predictions_array = np.array(predictions_list)
    mean_predictions = predictions_array.mean(axis=0)

    return mean_predictions