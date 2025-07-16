import numpy as np


def binomial_tree_option_price(S, K, T, r, sigma, N=100, option_type='call', american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros(N + 1)
    for i in range(N + 1):
        prices[i] = S * (u ** (N - i)) * (d ** i)

    values = np.maximum(0, (prices - K) if option_type == 'call' else (K - prices))

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            values[j] = np.exp(-r * dt) * (q * values[j] + (1 - q) * values[j + 1])
            if american:
                prices[j] = prices[j] / u
                intrinsic = max(0, prices[j] - K) if option_type == 'call' else max(0, K - prices[j])
                values[j] = max(values[j], intrinsic)

    return values[0]
