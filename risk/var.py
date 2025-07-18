from scipy.stats import norm
import numpy as np

def parametric_var(portfolio_returns, confidence_level=0.95):
    """
    Calculate Parametric VaR for portfolio returns.

    Parameters:
    - portfolio_returns: np.array or list, returns of the portfolio
    - confidence_level: float, confidence level (default 0.95)

    Returns:
    - VaR: float, the VaR value (loss)
    """
    mean = np.mean(portfolio_returns)
    std_dev = np.std(portfolio_returns)
    z = norm.ppf(1 - confidence_level)
    var = mean + std_dev * z
    return abs(var)  # positive number representing loss
