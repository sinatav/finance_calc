import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def autoregressive_model(data: list[float], lags: int):
    """Fit an AR model to the data."""
    model = AutoReg(data, lags=lags)
    model_fit = model.fit()
    return model_fit
