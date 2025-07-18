import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def moving_average_model(data: list[float], order: int):
    """Fit a MA model to the data."""
    model = ARIMA(data, order=(0, 0, order))
    model_fit = model.fit()
    return model_fit
