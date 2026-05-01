# src/models/transformer.py

"""
This model will:
    - Load a pretrained transformer (BERT, XLM-R, etc.)
    - Extract token embeddings
    - Apply mean pooling + max pooling
    - Concatenate them
    - Pass through classifier
"""


import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()


class TransformerModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        # Load pretrained transformer
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        # Regularization
        self.dropout = nn.Dropout(0.3)

        # Classification head
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def mean_pooling(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def max_pooling(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask == 0] = -1e9
        return torch.max(hidden_states, dim=1)[0]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        # Pooling
        mean_pool = self.mean_pooling(hidden_states, attention_mask)
        max_pool = self.max_pooling(hidden_states, attention_mask)

        # Combine
        pooled = torch.cat((mean_pool, max_pool), dim=1)

        x = self.dropout(pooled)
        logits = self.classifier(x)

        return logits