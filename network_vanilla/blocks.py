import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 50):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.model(x)
