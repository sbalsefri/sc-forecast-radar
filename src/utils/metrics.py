import numpy as np

EPS = 1e-8


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + EPS
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > EPS
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + EPS))))


def mase(y_true, y_pred, y_insample, seasonality=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_insample = np.asarray(y_insample)
    if len(y_insample) <= seasonality:
        return float("nan")
    naive = np.mean(np.abs(y_insample[seasonality:] - y_insample[:-seasonality])) + EPS
    return float(np.mean(np.abs(y_true - y_pred)) / naive)


METRIC_FUNCS = {
    "rmse": rmse,
    "mae": mae,
    "smape": smape,
    "mape": mape,
}
