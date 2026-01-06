import torch
from torch import nn
from .base import ForecastModel


class LSTMForecaster(ForecastModel):
    def __init__(self, context_length: int, horizon: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__(context_length, horizon)
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)
