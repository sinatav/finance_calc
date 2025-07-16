import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Option:
    """
    Black-Scholes European Option Pricing Model.

    Call price:  C = S N(d1) - K e^{-rT} N(d2)
    Put price:   P = K e^{-rT} N(-d2) - S N(-d1)

    where:
        d1 = [ln(S/K) + (r + σ²/2) T] / (σ sqrt(T))
        d2 = d1 - σ sqrt(T)

    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free interest rate
    sigma: Volatility (std dev)
    N(): CDF of standard normal distribution
    """
    ...

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S              # Stock price
        self.K = K              # Strike price
        self.T = T              # Time to maturity (in years)
        self.r = r              # Risk-free interest rate
        self.sigma = sigma      # Volatility (as decimal, e.g., 0.2)
        self.option_type = option_type.lower()

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        delta = norm.cdf(d1) if self.option_type == 'call' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2 if self.option_type == 'call' else -d2)) / 365
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T) / 100
        rho = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2 if self.option_type == 'call' else -d2)) / 100

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }



def plot_greeks_vs_price(K, T, r, sigma, option_type='call', S_range=(50, 150), step=1):
    """
    Plot option price and Greeks vs stock price.
    """
    S_vals = np.arange(S_range[0], S_range[1] + step, step)
    
    price_vals = []
    delta_vals = []
    gamma_vals = []
    theta_vals = []
    vega_vals = []
    rho_vals = []

    for S in S_vals:
        opt = Option(S, K, T, r, sigma, option_type)
        price_vals.append(opt.price())
        greeks = opt.greeks()
        delta_vals.append(greeks['Delta'])
        gamma_vals.append(greeks['Gamma'])
        theta_vals.append(greeks['Theta'])
        vega_vals.append(greeks['Vega'])
        rho_vals.append(greeks['Rho'])

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"{option_type.capitalize()} Option Greeks vs Stock Price", fontsize=16)

    axs[0, 0].plot(S_vals, price_vals, color='blue')
    axs[0, 0].set_title("Option Price")
    axs[0, 0].set_xlabel("Stock Price")
    axs[0, 0].set_ylabel("Price")

    axs[0, 1].plot(S_vals, delta_vals, color='green')
    axs[0, 1].set_title("Delta")
    axs[0, 1].set_xlabel("Stock Price")
    axs[0, 1].set_ylabel("Delta")

    axs[1, 0].plot(S_vals, gamma_vals, color='red')
    axs[1, 0].set_title("Gamma")
    axs[1, 0].set_xlabel("Stock Price")
    axs[1, 0].set_ylabel("Gamma")

    axs[1, 1].plot(S_vals, vega_vals, color='purple')
    axs[1, 1].set_title("Vega")
    axs[1, 1].set_xlabel("Stock Price")
    axs[1, 1].set_ylabel("Vega")

    axs[2, 0].plot(S_vals, theta_vals, color='orange')
    axs[2, 0].set_title("Theta")
    axs[2, 0].set_xlabel("Stock Price")
    axs[2, 0].set_ylabel("Theta")

    axs[2, 1].plot(S_vals, rho_vals, color='brown')
    axs[2, 1].set_title("Rho")
    axs[2, 1].set_xlabel("Stock Price")
    axs[2, 1].set_ylabel("Rho")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
