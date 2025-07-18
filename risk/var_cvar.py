import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_var_cvar(returns, confidence_level=0.95):
    mean = np.mean(returns)
    std_dev = np.std(returns)

    z = norm.ppf(1 - confidence_level)
    var = -(mean + z * std_dev)
    
    cvar = -mean + std_dev * norm.pdf(z) / (1 - confidence_level)

    return var, cvar

def visualize_var_cvar(returns, confidence_level=0.95):
    var, cvar = compute_var_cvar(returns, confidence_level)
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(-var, color='red', linestyle='--', label=f'VaR ({confidence_level*100:.0f}%): {-var:.2f}')
    plt.axvline(-cvar, color='orange', linestyle='--', label=f'CVaR: {-cvar:.2f}')
    plt.title(f'Value at Risk and Conditional VaR (Confidence Level: {confidence_level:.0%})')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
