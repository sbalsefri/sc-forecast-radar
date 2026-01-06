import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    context: int,
    horizon: int,
    val_windows: int = 1,
    test_windows: int = 1,
):
    series = []
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("t")
        y = g["y"].to_numpy(dtype=np.float32)

        total = len(y)
        needed = context + horizon
        if total < needed + (val_windows + test_windows - 1) * horizon:
            continue

        windows = []
        end = total
        for _ in range(test_windows):
            windows.append(("test", end))
            end -= horizon
        for _ in range(val_windows):
            windows.append(("val", end))
            end -= horizon
        windows = list(reversed(windows))

        train_end = windows[0][1] - horizon

        Xtr, ytr = [], []
        for i in range(context, train_end - horizon + 1):
            Xtr.append(y[i - context : i])
            ytr.append(y[i : i + horizon])

        if not Xtr:
            continue

        Xv, yv, Xt, yt = [], [], [], []
        for split, end_idx in windows:
            x = y[end_idx - horizon - context : end_idx - horizon]
            target = y[end_idx - horizon : end_idx]
            if split == "val":
                Xv.append(x)
                yv.append(target)
            else:
                Xt.append(x)
                yt.append(target)

        series.append(
            (
                sid,
                y,
                np.stack(Xtr),
                np.stack(ytr),
                np.stack(Xv) if Xv else None,
                np.stack(yv) if yv else None,
                np.stack(Xt),
                np.stack(yt),
            )
        )

    if not series:
        raise RuntimeError("No series long enough for windowing")

    X_train = np.concatenate([s[2] for s in series])
    y_train = np.concatenate([s[3] for s in series])

    X_val = (
        np.concatenate([s[4] for s in series if s[4] is not None])
        if any(s[4] is not None for s in series)
        else None
    )
    y_val = (
        np.concatenate([s[5] for s in series if s[5] is not None])
        if any(s[5] is not None for s in series)
        else None
    )

    X_test = np.concatenate([s[6] for s in series])
    y_test = np.concatenate([s[7] for s in series])

    insample = {sid: y for sid, y, *_ in series}

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "insample": insample,
    }
