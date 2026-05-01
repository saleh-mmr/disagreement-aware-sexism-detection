# src/losses/losses.py
import torch
import torch.nn as nn

"""
Update Loss Function - Now we use: KL Divergence Loss
The KL Divergence Loss is a measure of how
one probability distribution diverges from a second,
expected probability distribution. In the context of machine learning,
it is often used to compare the predicted probability distribution (logits)
with the true probability distribution (targets).
"""
class SoftLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        return self.kl(log_probs, targets)