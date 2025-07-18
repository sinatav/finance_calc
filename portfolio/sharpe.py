def sharpe_ratio(expected_return, risk_free_rate, portfolio_std_dev):
    """
    Calculate Sharpe Ratio.

    Parameters:
    - expected_return: float, expected portfolio return
    - risk_free_rate: float, risk-free rate
    - portfolio_std_dev: float, portfolio standard deviation (risk)

    Returns:
    - float: Sharpe Ratio
    """
    excess_return = expected_return - risk_free_rate
    return excess_return / portfolio_std_dev if portfolio_std_dev != 0 else float('inf')
