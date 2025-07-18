import numpy as np

def portfolio_variance(weights, covariance_matrix):
    """
    Calculate portfolio variance for n assets.

    Parameters:
    - weights: list or np.array of portfolio weights (must sum to 1)
    - covariance_matrix: 2D list or np.array, covariance matrix of asset returns

    Returns:
    - float, portfolio variance
    """
    weights = np.array(weights)
    covariance_matrix = np.array(covariance_matrix)

    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D list or array.")
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if len(weights) != covariance_matrix.shape[0]:
        raise ValueError("Weights and covariance matrix size mismatch.")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Sum of weights must be 1.")

    return float(weights.T @ covariance_matrix @ weights)
