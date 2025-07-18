def capital_market_line(rf, rm, sigma_m, sigma_c):
    """
    Calculate expected return on the Capital Market Line.

    Parameters:
    - rf: Risk-free rate
    - rm: Expected return of the market portfolio
    - sigma_m: Standard deviation of the market portfolio
    - sigma_c: Standard deviation of the combined portfolio

    Returns:
    - float, expected return of the portfolio on the CML
    """
    return rf + ((rm - rf) / sigma_m) * sigma_c
