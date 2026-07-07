# src/models/structure_model.py
import torch
import torch.nn as nn

class GraphMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        """
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        """

        self.net=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.net(x)  