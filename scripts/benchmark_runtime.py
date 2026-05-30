import os
import time
import torch
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running this
# script directly from the repository (e.g. `python scripts/benchmark_runtime.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from src.config import (
    MODEL_NAMES,
    NUM_CLASSES,
    MAX_LEN,
    BATCH_SIZE,
    DEVICE,
    MODEL_OUTPUT,
    DEV_PATH,
    MODE
)
from src.data.preprocessing import load_annotated_data
from src.data.dataset import SexismDataset
from src.models.transformer import TransformerModel
from src.engine.predictor import predict_probabilities


OUTPUT_DIR = "outputs/analysis"


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def get_checkpoint_size_mb(path):
    if not os.path.exists(path):
        return None

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def load_trained_model(model_name):
    model = TransformerModel(model_name, NUM_CLASSES)

    model_path = MODEL_OUTPUT + f"{model_name.replace('/', '_')}_{MODE}.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Run python train.py first."
        )

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model, model_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading development data...")
    dev_df = load_annotated_data(DEV_PATH)

    results = []

    for model_name in MODEL_NAMES:
        print("\n==============================")
        print(f"Benchmarking model: {model_name}")
        print("==============================")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        dev_dataset = SexismDataset(
            dev_df["text"].values,
            None,
            tokenizer,
            MAX_LEN
        )

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model, model_path = load_trained_model(model_name)

        total_params, trainable_params = count_parameters(model)
        checkpoint_size_mb = get_checkpoint_size_mb(model_path)

        # Warm-up pass.
        print("Running warm-up inference...")
        _ = predict_probabilities(model, dev_loader, DEVICE)

        print("Running timed inference...")
        start_time = time.time()

        _ = predict_probabilities(model, dev_loader, DEVICE)

        end_time = time.time()

        total_time_seconds = end_time - start_time
        num_examples = len(dev_df)
        avg_time_per_example = total_time_seconds / num_examples
        examples_per_second = num_examples / total_time_seconds

        results.append({
            "model_name": model_name,
            "device": DEVICE,
            "num_examples": num_examples,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_path": model_path,
            "checkpoint_size_mb": checkpoint_size_mb,
            "total_inference_time_seconds": total_time_seconds,
            "average_time_per_example_seconds": avg_time_per_example,
            "examples_per_second": examples_per_second
        })

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")
        print(f"Total inference time: {total_time_seconds:.2f} seconds")
        print(f"Average time per example: {avg_time_per_example:.6f} seconds")
        print(f"Examples per second: {examples_per_second:.2f}")

    results_df = pd.DataFrame(results)

    output_path = os.path.join(
        OUTPUT_DIR,
        "runtime_benchmark.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n===== Runtime Benchmark Summary =====")
    print(results_df.to_string(index=False))

    print("\nSaved file:")
    print(output_path)


if __name__ == "__main__":
    main()