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

def block_missing(x: np.ndarray, block_size: int, rng: np.random.Generator):
    """
    Contiguous block missingness (e.g., sensor outage).
    Zeros out a random contiguous block in each sample.
    """
    if block_size <= 0:
        return x

    x2 = x.copy()
    T = x2.shape[1]

    for i in range(x2.shape[0]):
        if T <= block_size:
            continue
        start = rng.integers(0, T - block_size)
        x2[i, start : start + block_size] = 0.0

    return x2


def delay_shift(x: np.ndarray, max_delay: int, rng: np.random.Generator):
    """
    Random reporting / logistics delay.
    Shifts the series forward and pads with zeros.
    """
    if max_delay <= 0:
        return x

    x2 = x.copy()
    T = x2.shape[1]

    for i in range(x2.shape[0]):
        d = rng.integers(1, max_delay + 1)
        if d >= T:
            continue
        x2[i, d:] = x2[i, :-d]
        x2[i, :d] = 0.0

    return x2

