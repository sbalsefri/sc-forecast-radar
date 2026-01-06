import torch
from torch import nn
from .base import ForecastModel


class PatchTSTLite(ForecastModel):
    """
    Lightweight PatchTST-style forecaster
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        patch_len: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(context_length, horizon)

        self.patch_len = patch_len
        self.n_patches = (context_length + patch_len - 1) // patch_len

        self.proj = nn.Linear(patch_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape

        pad = self.n_patches * self.patch_len - L
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)

        x = x.view(B, self.n_patches, self.patch_len)
        z = self.proj(x)
        z = self.encoder(z)
        z = z.mean(dim=1)

        return self.head(z)
