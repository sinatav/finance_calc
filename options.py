import numpy as np
from scipy.stats import norm

class Option:
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
