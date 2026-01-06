import torch
from torch import nn
from .base import ForecastModel


class DLinear(ForecastModel):
    def __init__(self, context_length: int, horizon: int, kernel: int = 7):
        super().__init__(context_length, horizon)
        self.kernel = kernel
        self.lin_trend = nn.Linear(context_length, horizon)
        self.lin_seas = nn.Linear(context_length, horizon)

    def moving_avg(self, x):
        pad = self.kernel // 2
        xpad = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
        w = torch.ones(1, 1, self.kernel, device=x.device) / self.kernel
        ma = torch.nn.functional.conv1d(xpad.unsqueeze(1), w, padding=0).squeeze(1)
        return ma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_avg(x)
        seas = x - trend
        return self.lin_trend(trend) + self.lin_seas(seas)
