import os
import torch
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAMES,
    NUM_CLASSES,
    MAX_LEN,
    DEVICE,
    MODEL_OUTPUT,
    MODE
)
from src.models.transformer import TransformerModel


CLASS_NAMES = ["NO", "YES"]


class TransformerLimeWrapper:
    """
    Wrapper that allows LIME to call a PyTorch transformer model.

    LIME passes a list of raw text strings and expects a NumPy array
    of prediction probabilities with shape:

        [num_examples, num_classes]
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
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

        return model

    def predict_proba(self, texts):
        encodings = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(DEVICE)
        attention_mask = encodings["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()


def explain_text(model_name, text, num_features=10):
    wrapper = TransformerLimeWrapper(model_name)

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES
    )

    probs = wrapper.predict_proba([text])[0]
    predicted_label = int(np.argmax(probs))

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=wrapper.predict_proba,
        num_features=num_features,
        labels=[predicted_label]
    )

    return explanation


def save_explanation_to_csv(explanation, output_path):
    rows = []

    for label_idx in explanation.available_labels():
        label_name = CLASS_NAMES[label_idx]

        for token, weight in explanation.as_list(label=label_idx):
            rows.append({
                "class": label_name,
                "token": token,
                "weight": weight
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def save_explanation_to_html(explanation, output_path):
    html = explanation.as_html()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    os.makedirs("outputs/explainability", exist_ok=True)

    # You can replace these examples with interesting examples
    # from outputs/analysis/dev_high_confidence_errors_top50.csv
    # or outputs/analysis/dev_low_margin_errors_top50.csv.
    examples = [
        {
            "name": "example_clear_non_sexist",
            "text": "This article discusses online harassment and how platforms should protect users."
        },
        {
            "name": "example_ambiguous",
            "text": "Sure, women are obviously great leaders, that's why everything is going so well."
        },
        {
            "name": "example_clear_sexist",
            "text": "I hate women should be 1000% percent submissive men and pick me ass bitches",
        }
    ]

    # Use the first model by default.
    # You can also change this to "bert-base-multilingual-cased".
    model_name = MODEL_NAMES[0]

    for example in examples:
        print(f"\nExplaining: {example['name']}")
        print(example["text"])

        explanation = explain_text(
            model_name=model_name,
            text=example["text"],
            num_features=10
        )

        csv_path = os.path.join(
            "outputs/explainability",
            f"{example['name']}_lime.csv"
        )

        html_path = os.path.join(
            "outputs/explainability",
            f"{example['name']}_lime.html"
        )

        save_explanation_to_csv(explanation, csv_path)
        save_explanation_to_html(explanation, html_path)

        print(f"Saved CSV: {csv_path}")
        print(f"Saved HTML: {html_path}")


if __name__ == "__main__":
    main()