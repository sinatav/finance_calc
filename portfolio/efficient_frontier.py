import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, returns):
    return weights.T @ returns

def efficient_frontier(returns, cov_matrix, num_portfolios=100):
    """
    Plot efficient frontier for given asset returns and covariance matrix.

    Parameters:
    - returns: np.array, expected returns of assets
    - cov_matrix: np.array, covariance matrix
    - num_portfolios: int, number of portfolios to simulate

    Returns:
    - None (plots frontier)
    """
    n = len(returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    target_returns = np.linspace(min(returns), max(returns), num_portfolios)

    for i, target in enumerate(target_returns):
        # Minimize variance for target return
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target}
        )
        result = minimize(portfolio_variance, n * [1. / n], args=(cov_matrix,), bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x
            weights_record.append(weights)
            results[0, i] = portfolio_variance(weights, cov_matrix) ** 0.5  # std dev (risk)
            results[1, i] = target  # expected return
            results[2, i] = (results[1, i] / results[0, i]) if results[0, i] != 0 else 0  # Sharpe Ratio approx
        else:
            raise RuntimeError("Optimization failed for target return")

    plt.figure(figsize=(10, 6))
    plt.plot(results[0, :], results[1, :], 'b-', label='Efficient Frontier')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
