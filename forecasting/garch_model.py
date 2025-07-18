import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_model(returns):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    forecast = model_fit.forecast(horizon=5)

    print(model_fit.summary())
    plt.plot(forecast.variance[-1:], label="Forecasted Variance")
    plt.title("GARCH Forecast")
    plt.legend()
    plt.show()