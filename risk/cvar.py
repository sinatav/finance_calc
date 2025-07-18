import numpy as np

def conditional_var(portfolio_returns, confidence_level=0.95):
    """
    Calculate Conditional VaR (CVaR), aka Expected Shortfall.

    Parameters:
    - portfolio_returns: np.array or list, returns of the portfolio
    - confidence_level: float, confidence level (default 0.95)

    Returns:
    - CVaR: float, expected shortfall beyond VaR
    """
    var = parametric_var(portfolio_returns, confidence_level)
    losses = portfolio_returns[portfolio_returns <= -var]
    if len(losses) == 0:
        return var  # fallback if no losses beyond VaR in sample
    return abs(np.mean(losses))
