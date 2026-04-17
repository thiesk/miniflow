import torch
import torch.nn as nn

class Miniflow(torch.nn.Module):
    def __init__(self, config):
        super(Miniflow, self).__init__()
        self.config = config
        hdim = self.config["model"]["hidden_dim"]
        self.vel_t= nn.Sequential(
            nn.Linear(3,hdim),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hdim, 2)
        )

    def forward(self, x):
        return self.vel_t(x)

