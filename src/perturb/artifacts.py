import numpy as np


def missing_mcar(x, rate, rng):
    if rate <= 0:
        return x
    x2 = x.copy()
    mask = rng.random(x2.shape) < rate
    x2[mask] = 0.0
    return x2


def outlier_spike(x, rate, rng, mag=5.0):
    if rate <= 0:
        return x
    x2 = x.copy()
    mask = rng.random(x2.shape) < rate
    x2[mask] = x2[mask] * (1 + mag * rng.random(mask.sum()))
    return x2
