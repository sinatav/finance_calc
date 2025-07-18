def capital_asset_pricing_model(rf, beta, rm):
    """
    Calculate expected return using CAPM.

    Parameters:
    - rf: Risk-free rate
    - beta: Beta of the asset
    - rm: Expected market return

    Returns:
    - Expected return of the asset
    """
    return rf + beta * (rm - rf)
