import numpy as np


def missing_mcar(x: np.ndarray, rate: float, rng: np.random.Generator):
    """
    Missing Completely At Random
    """
    if rate <= 0:
        return x
    x2 = x.copy()
    mask = rng.random(x2.shape) < rate
    x2[mask] = 0.0
    return x2


def outlier_spike(x: np.ndarray, rate: float, rng: np.random.Generator, magnitude: float = 5.0):
    """
    Promotional spikes / abnormal values
    """
    if rate <= 0:
        return x
    x2 = x.copy()
    mask = rng.random(x2.shape) < rate
    x2[mask] = x2[mask] * (1 + magnitude * rng.random(mask.sum()))
    return x2
