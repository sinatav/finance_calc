import numpy as np


def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', num_simulations=100000):
    np.random.seed(42)
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type")

    price = np.exp(-r * T) * np.mean(payoffs)
    return price
