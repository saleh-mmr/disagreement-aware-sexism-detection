# predict_test.py

import os
import torch
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAMES,
    NUM_CLASSES,
    MAX_LEN,
    BATCH_SIZE,
    DEVICE,
    MODEL_OUTPUT,
    TEST_PATH,
    PREDICTION_OUTPUT,
    MODE
)
from src.data.preprocessing import load_test_data
from src.data.dataset import SexismDataset
from src.models.transformer import TransformerModel
from src.engine.predictor import predict_probabilities, ensemble_mean
from src.engine.prediction_writer import write_task1_predictions


def load_trained_model(model_name):
    """
    Load one trained Transformer model from disk.
    """
    model = TransformerModel(model_name, NUM_CLASSES)

    model_path = MODEL_OUTPUT + f"{model_name.replace('/', '_')}_{MODE}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Run python train.py first."
        )

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def main():
    os.makedirs(PREDICTION_OUTPUT, exist_ok=True)
    
    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)

    all_predictions = []

    for model_name in MODEL_NAMES:
        print("\n==============================")
        print(f"Predicting with model: {model_name}")
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        test_dataset = SexismDataset(
            test_df["text"].values,
            None,
            tokenizer,
            MAX_LEN
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = load_trained_model(model_name)

        preds = predict_probabilities(model, test_loader, DEVICE)
        all_predictions.append(preds)

    print("\n===== ENSEMBLE TEST PREDICTIONS =====")

    ensemble_preds = ensemble_mean(all_predictions)

    test_prediction_path = PREDICTION_OUTPUT + "test_task1_ensemble_predictions.json"

    write_task1_predictions(
        ids=test_df["id_EXIST"].values,
        probabilities=ensemble_preds,
        output_path=test_prediction_path
    )

    print("Done.")


if __name__ == "__main__":
    main()