# src/data/dataset.py
import torch

"""
Now labels will be vectors, not integers.
to follow the project rule.
"""
class SexismDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)  # now list of vectors
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # Replace the labels line in __getitem__ with:
            "labels": torch.tensor(label, dtype=torch.float if isinstance(label, list) else torch.long)
        }