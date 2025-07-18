import numpy as np

def autoregressive_model(data: list[float], coeffs: list[float], c: float = 0.0) -> np.ndarray:
    """
    AR(p) model prediction: X_t = c + sum(phi_i * X_{t-i}) + noise
    coeffs = [phi_1, phi_2, ..., phi_p]
    Returns the predicted series (for simplicity assumes zero noise)
    """
    p = len(coeffs)
    data = np.array(data)
    n = len(data)
    pred = np.zeros(n)
    for t in range(p, n):
        pred[t] = c + sum(coeffs[i] * data[t - i - 1] for i in range(p))
    return pred
