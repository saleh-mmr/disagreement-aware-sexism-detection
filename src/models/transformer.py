# src/models/transformer.py

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
"""
This model will:
    - Load a pretrained transformer (BERT, XLM-R, etc.)
    - Extract token embeddings
    - Apply mean pooling + max pooling
    - Concatenate them
    - Pass through classifier
"""

class TransformerModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        # Load pretrained (Multilingual) transformer
        # Could be 'bert-base-multilingual-cased', 'xlm-roberta-base'.
        # Because "Dataset EXIST" contains English and Spanish tweets.
        self.encoder = AutoModel.from_pretrained(model_name)

        # Get hidden size from the encoder config
        # For BERT-base and XLM-R-base, this is usually 768.
        hidden_size = self.encoder.config.hidden_size

        # Regularization
        # Dropout helps reduce overfitting by randomly disabling some neurons during training.
        self.dropout = nn.Dropout(0.3)

        # Classification head
        # We concatenate mean and max pooled features, so input size is hidden_size * 2.
        self.classifier = nn.Linear(hidden_size * 2, num_classes)


# ******* Important technical note ********
# hidden_states[mask == 0] = -1e9
# This line modifies hidden_states in-place
# in-place modification can sometimes create gradient/autograd issues.

    def mean_pooling(self, hidden_states, attention_mask):
        """
        This averages token embeddings across the sentence, ignoring padding tokens.
        The transformer returns one vector per token. Mean pooling combines them into one sentence-level vector.
        """
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def max_pooling(self, hidden_states, attention_mask):
        """
        This takes the strongest feature value across tokens.
        """
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[mask == 0] = -1e9
        return torch.max(hidden_states, dim=1)[0]
    
    # model uses two views of the tweet:
    # 1. Mean pooling: captures the overall sentiment by averaging token embeddings.
    # 2. Max pooling: captures the most salient features by taking the maximum value across
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