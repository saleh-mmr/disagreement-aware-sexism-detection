# src/data/dataset.py

import torch


class SexismDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for EXIST Task 1.

    Supports:
    - training/dev with labels
    - test prediction without labels
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

        if self.labels is not None:
            label = self.labels[idx]

            item["labels"] = torch.tensor(
                label,
                dtype=torch.float if isinstance(label, list) else torch.long
            )

        return item