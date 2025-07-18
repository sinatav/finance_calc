from statsmodels.tsa.arima.model import ARIMA

def fit_arima(data: list[float], order: tuple) -> ARIMA:
    """
    Fit ARIMA(p,d,q) model.
    order: (p, d, q)
    """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit
