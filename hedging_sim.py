import numpy as np
import matplotlib.pyplot as plt
from options import Option

def simulate_delta_hedging(S0, K, T, r, sigma, option_type='call', steps=50):
    dt = T / steps
    prices = [S0]
    np.random.seed(42)

    for _ in range(steps):
        z = np.random.normal()
        St = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(St)

    portfolio_values = []
    cash_positions = []
    shares_held = 0
    cash = 0

    for i, t_step in enumerate(np.linspace(T, 0, steps + 1)):
        S = prices[i]
        opt = Option(S, K, t_step, r, sigma, option_type)
        delta = opt.greeks()['Delta']

        if i == 0:
            shares_held = delta
            cash = opt.price() - shares_held * S
        else:
            d_shares = delta - shares_held
            cash -= d_shares * S
            shares_held = delta

        portfolio_val = shares_held * S + cash - opt.price() if t_step > 0 else shares_held * S + cash - max(0, S - K if option_type=='call' else K - S)
        portfolio_values.append(portfolio_val)
        cash_positions.append(cash)

    plt.plot(portfolio_values, label='Hedged Portfolio Value')
    plt.axhline(0, color='red', linestyle='--', label='Break-even')
    plt.title('Delta Hedging Simulation')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

def simulate_covered_call(S0, K, T, r, sigma, steps=50):
    dt = T / steps
    prices = [S0]
    np.random.seed(42)

    for _ in range(steps):
        z = np.random.normal()
        St = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(St)

    opt = Option(S0, K, T, r, sigma, 'call')
    option_price = opt.price()
    portfolio_values = []

    for i, t_step in enumerate(np.linspace(T, 0, steps + 1)):
        S = prices[i]
        time_to_exp = max(t_step, 0.0001)
        opt = Option(S, K, time_to_exp, r, sigma, 'call')
        option_price = opt.price()

        portfolio_val = S - option_price + option_price
        portfolio_values.append(portfolio_val)

    plt.plot(portfolio_values, label='Covered Call Portfolio Value')
    plt.title('Covered Call Strategy Simulation')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
