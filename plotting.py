import numpy as np
import matplotlib.pyplot as plt
from options import Option
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm


def plot_delta_gamma_surface(K, r, sigma, option_type='call', S_range=(50, 150), T_range=(0.01, 1), steps=50):
    """
    Plot 3D surface of Delta and Gamma vs Stock Price and Time to Maturity.
    """
    S_vals = np.linspace(S_range[0], S_range[1], steps)
    T_vals = np.linspace(T_range[0], T_range[1], steps)

    S_grid, T_grid = np.meshgrid(S_vals, T_vals)
    delta_grid = np.zeros_like(S_grid)
    gamma_grid = np.zeros_like(S_grid)

    for i in range(steps):
        for j in range(steps):
            opt = Option(S_grid[i, j], K, T_grid[i, j], r, sigma, option_type)
            greeks = opt.greeks()
            delta_grid[i, j] = greeks['Delta']
            gamma_grid[i, j] = greeks['Gamma']

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(S_grid, T_grid, delta_grid, cmap=cm.viridis)
    ax1.set_title('Delta Surface')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Delta')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(S_grid, T_grid, gamma_grid, cmap=cm.plasma)
    ax2.set_title('Gamma Surface')
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Time to Maturity')
    ax2.set_zlabel('Gamma')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.suptitle(f'{option_type.capitalize()} Option Greeks Surfaces')
    plt.show()


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
