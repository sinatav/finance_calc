import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StressTestSimulator:
    def __init__(self, portfolio_returns):
        self.returns = portfolio_returns

    def simulate_shock(self, shock_percentage):
        """Apply a market shock to portfolio returns."""
        shocked_returns = self.returns * (1 + shock_percentage)
        return shocked_returns

    def run_stress_scenario(self, shock_scenarios):
        """Simulate and summarize portfolio under different shock scenarios."""
        results = {}
        for label, shock in shock_scenarios.items():
            shocked = self.simulate_shock(shock)
            results[label] = {
                "mean_return": np.mean(shocked),
                "std_dev": np.std(shocked),
                "min_return": np.min(shocked),
                "max_return": np.max(shocked),
            }
        return pd.DataFrame(results).T

    def plot_shock_effects(self, shock_scenarios):
        """Plot histogram under stress scenarios."""
        plt.figure(figsize=(10, 6))
        for label, shock in shock_scenarios.items():
            shocked = self.simulate_shock(shock)
            plt.hist(shocked, bins=50, alpha=0.5, label=f"{label} ({shock*100:.0f}%)")
        plt.title("Portfolio Return Distributions Under Stress Scenarios")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
