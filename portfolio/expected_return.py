def expected_portfolio_return(weights, expected_returns):
    """
    Calculates the expected return of a portfolio.

    Parameters:
    - weights: list of floats, weights of assets (must sum to 1)
    - expected_returns: list of floats, expected returns of each asset

    Returns:
    - float, expected return of the portfolio
    """
    if len(weights) != len(expected_returns):
        raise ValueError("Weights and expected returns must have the same length.")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Sum of weights must be 1.")

    return sum(w * r for w, r in zip(weights, expected_returns))
