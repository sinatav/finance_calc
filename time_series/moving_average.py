import numpy as np

def moving_average(data: list[float], window: int) -> np.ndarray:
    """
    Simple Moving Average (SMA) over the window size.
    """
    data = np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def exponential_moving_average(data: list[float], alpha: float) -> np.ndarray:
    """
    Exponential Moving Average (EMA).
    alpha: smoothing factor between 0 and 1.
    """
    data = np.array(data)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema
