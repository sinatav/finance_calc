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


    def delta_hedge_shares(self, target_exposure=1):
        """
        Calculate number of shares of underlying to hedge the option delta.

        target_exposure: number of option contracts (default 1)
        Returns:
            shares needed to hedge delta exposure (can be fractional)
        """
        delta = self.greeks()['Delta']
        # Hedge shares = -delta * number of options (negative delta means short shares)
        return -delta * target_exposure

    def gamma_hedge_options(self, other_option, target_exposure=1):
        """
        Calculate how many of another option are needed to gamma hedge this option.

        other_option: an Option object with different strike/maturity
        target_exposure: number of this option contracts (default 1)
        Returns:
            number of other_option contracts to hedge gamma risk
        """
        gamma_self = self.greeks()['Gamma']
        gamma_other = other_option.greeks()['Gamma']
        if gamma_other == 0:
            raise ValueError("Cannot hedge with option having zero gamma")
        return - (gamma_self * target_exposure) / gamma_other

    def theta_vega_positioning(self, other_option, target_exposure=1):
        """
        Calculate ratio of other_option contracts needed to offset Theta and Vega risk.

        other_option: an Option object with different strike/maturity
        target_exposure: number of this option contracts (default 1)
        Returns:
            dict with number of other_option to hedge Theta and Vega separately
        """
        theta_self = self.greeks()['Theta']
        vega_self = self.greeks()['Vega']
        theta_other = other_option.greeks()['Theta']
        vega_other = other_option.greeks()['Vega']

        results = {}
        if theta_other != 0:
            results['theta_hedge'] = - (theta_self * target_exposure) / theta_other
        else:
            results['theta_hedge'] = None

        if vega_other != 0:
            results['vega_hedge'] = - (vega_self * target_exposure) / vega_other
        else:
            results['vega_hedge'] = None

        return results


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
