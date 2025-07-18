import numpy as np

def calculate_beta(asset_returns, market_returns):
    """
    Calculate beta of an asset relative to the market.

    Parameters:
    - asset_returns: array-like, returns of the asset
    - market_returns: array-like, returns of the market

    Returns:
    - beta: float, market risk measure
    """
    asset_returns = np.array(asset_returns)
    market_returns = np.array(market_returns)

    covariance = np.cov(asset_returns, market_returns, ddof=1)[0, 1]
    variance_market = np.var(market_returns, ddof=1)

    beta = covariance / variance_market
    return beta
