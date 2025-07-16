import numpy as np


def crank_nicolson_option_price(S, K, T, r, sigma, Smax=200, M=100, N=1000, option_type='call'):
    dS = Smax / M
    dt = T / N
    i = np.arange(1, M)
    alpha = 0.25 * dt * ((sigma ** 2) * (i ** 2) - r * i)
    beta = -dt * 0.5 * ((sigma ** 2) * (i ** 2) + r)
    gamma = 0.25 * dt * ((sigma ** 2) * (i ** 2) + r * i)

    A = np.diag(1 - beta) + np.diag(-alpha[1:], -1) + np.diag(-gamma[:-1], 1)
    B = np.diag(1 + beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)

    grid = np.zeros((N + 1, M + 1))
    grid[-1, 1:M] = np.maximum(0, (i * dS - K) if option_type == 'call' else (K - i * dS))

    from scipy.linalg import solve_banded
    for j in reversed(range(N)):
        b = B @ grid[j + 1, 1:M]
        grid[j, 1:M] = np.linalg.solve(A, b)

    S_index = int(S / dS)
    return np.interp(S, np.arange(0, Smax + dS, dS), grid[0])
