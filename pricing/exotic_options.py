import numpy as np


def asian_option_price(S, K, T, r, sigma, option_type='call', n_sim=100000, n_steps=100):
    dt = T / n_steps
    discount = np.exp(-r * T)
    np.random.seed(42)

    payoffs = []
    for _ in range(n_sim):
        prices = [S]
        for _ in range(n_steps):
            z = np.random.normal()
            S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            prices.append(S_t)
        avg_price = np.mean(prices)
        if option_type == 'call':
            payoff = max(avg_price - K, 0)
        else:
            payoff = max(K - avg_price, 0)
        payoffs.append(payoff)

    return discount * np.mean(payoffs)


def digital_option_price(S, K, T, r, sigma, option_type='call', n_sim=100000):
    np.random.seed(42)
    z = np.random.normal(size=n_sim)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    discount = np.exp(-r * T)

    if option_type == 'call':
        payoff = np.where(ST > K, 1, 0)
    else:
        payoff = np.where(ST < K, 1, 0)

    return discount * np.mean(payoff)
