import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_model(data: list[float], p: int, d: int, q: int):
    """Fit an ARIMA model to the data."""
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit
