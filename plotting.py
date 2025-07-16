import numpy as np
import matplotlib.pyplot as plt
from options import Option

def plot_greeks_vs_time(S, K, r, sigma, option_type='call', T_max=1, steps=100):
    """
    Plot Greeks evolution as time to maturity approaches zero.
    
    Parameters:
        S: Stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        T_max: Initial time to maturity (years)
        steps: Number of time steps until expiration
    """
    times = np.linspace(T_max, 0.001, steps)  # Avoid zero division
    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []

    for T in times:
        opt = Option(S, K, T, r, sigma, option_type)
        greeks = opt.greeks()
        deltas.append(greeks['Delta'])
        gammas.append(greeks['Gamma'])
        thetas.append(greeks['Theta'])
        vegas.append(greeks['Vega'])
        rhos.append(greeks['Rho'])

    plt.figure(figsize=(10, 6))
    plt.plot(times, deltas, label='Delta')
    plt.plot(times, gammas, label='Gamma')
    plt.plot(times, thetas, label='Theta')
    plt.plot(times, vegas, label='Vega')
    plt.plot(times, rhos, label='Rho')
    plt.gca().invert_xaxis()  # Show time approaching zero from left to right
    plt.title(f'{option_type.capitalize()} Option Greeks Evolution vs Time to Maturity')
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Greek Value')
    plt.legend()
    plt.grid(True)
    plt.show()
