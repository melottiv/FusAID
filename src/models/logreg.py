# src/models/logreg.py
import torch
import torch.nn as nn

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.classifier(x)  # logits, da passare a BCEWithLogitsLoss