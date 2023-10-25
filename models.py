# pytorch model definitions to be used throughout notebooks in the repo:

import torch
from torch import nn

class LinearRegressionModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=1,
                                       out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer1(x)