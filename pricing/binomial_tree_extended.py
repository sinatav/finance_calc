def one_step_binomial_call(S, K, u, d, r):
    """
    One-step binomial call option pricing.
    """
    # Terminal payoffs
    Cu = max(0, S * u - K)
    Cd = max(0, S * d - K)

    # Risk-neutral probability
    p = (1 + r - d) / (u - d)

    # Present value
    C = (1 / (1 + r)) * (p * Cu + (1 - p) * Cd)

    return {
        "Call Price": C,
        "Cu": Cu,
        "Cd": Cd,
        "p": p
    }


def multi_step_binomial_call(S, K, T, r, sigma, N):
    """
    Multi-step binomial option pricing for European call option.
    """
    dt = T / N
    u = 1 + sigma * (dt ** 0.5)
    d = 1 - sigma * (dt ** 0.5)
    p = (1 + r * dt - d) / (u - d)

    # Build asset prices at maturity
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    # Option values at maturity
    option_values = [max(0, price - K) for price in asset_prices]

    # Work backward through tree
    for i in reversed(range(N)):
        for j in range(i + 1):
            option_values[j] = (p * option_values[j + 1] + (1 - p) * option_values[j]) / (1 + r * dt)

    return option_values[0]
