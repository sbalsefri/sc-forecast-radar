import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_torch(
    model: nn.Module,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(epochs), desc="training"):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    return model
