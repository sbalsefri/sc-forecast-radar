import torch
from torch import nn
from .base import ForecastModel


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., : x.shape[-1]]
        return y + self.down(x)


class TCNForecaster(ForecastModel):
    def __init__(self, context_length: int, horizon: int, channels=64, levels=4, dropout=0.1):
        super().__init__(context_length, horizon)
        blocks = []
        in_ch = 1
        for i in range(levels):
            d = 2 ** i
            blocks.append(TemporalBlock(in_ch, channels, k=3, d=d, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        y = self.tcn(x)
        return self.head(y)

