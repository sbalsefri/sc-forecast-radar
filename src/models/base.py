from torch import nn
import torch


class ForecastModel(nn.Module):
    def __init__(self, context_length: int, horizon: int):
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
