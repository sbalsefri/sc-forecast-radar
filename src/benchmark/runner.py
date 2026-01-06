import numpy as np
from pathlib import Path

from src.utils.seed import set_seed
from src.utils.metrics import METRIC_FUNCS, mase
from src.data.generic import load_generic_csv
from src.data.windows import make_windows
from src.models.dlinear import DLinear
from src.train.trainer import train_torch
from src.perturb.artifacts import missing_mcar, outlier_spike


def run_benchmark(
    csv_path: str,
    id_col: str,
    time_col: str,
    target_col: str,
    context: int = 24,
    horizon: int = 12,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
):
    set_seed(seed)

    df = load_generic_csv(csv_path, id_col, time_col, target_col)
    windows = make_windows(df, context, horizon)

    X_train = windows["X_train"]
    y_train = windows["y_train"]
    X_test = windows["X_test"]
    y_test = windows["y_test"]
    insample = np.concatenate(list(windows["insample"].values()))

    model = DLinear(context, horizon)
    model = train_torch(
        model,
        X_train,
        y_train,
        None,
        None,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
    )

    rng = np.random.default_rng(seed)

    results = []

    # clean
    pred = model(
        np.asarray(X_test, dtype=np.float32)
    ).detach().cpu().numpy()
    row = {"perturbation": "clean"}
    for m, fn in METRIC_FUNCS.items():
        row[m] = fn(y_test.flatten(), pred.flatten())
    row["mase"] = mase(y_test.flatten(), pred.flatten(), insample)
    results.append(row)

    # missing
    Xm = missing_mcar(X_test, 0.1, rng)
    pred = model(
        np.asarray(Xm, dtype=np.float32)
    ).detach().cpu().numpy()
    row = {"perturbation": "missing_mcar"}
    for m, fn in METRIC_FUNCS.items():
        row[m] = fn(y_test.flatten(), pred.flatten())
    row["mase"] = mase(y_test.flatten(), pred.flatten(), insample)
    results.append(row)

    # outliers
    Xo = outlier_spike(X_test, 0.05, rng)
    pred = model(
        np.asarray(Xo, dtype=np.float32)
    ).detach().cpu().numpy()
    row = {"perturbation": "outlier_spike"}
    for m, fn in METRIC_FUNCS.items():
        row[m] = fn(y_test.flatten(), pred.flatten())
    row["mase"] = mase(y_test.flatten(), pred.flatten(), insample)
    results.append(row)

    return results
