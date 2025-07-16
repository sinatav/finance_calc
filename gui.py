import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QTabWidget, QGridLayout, QComboBox, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from interests import SimpleInterest, CompoundInterest, ContinuousCompounding
from bonds import ZeroCouponBond, CouponBearingBond
from options import Option, plot_greeks_vs_price  # Make sure plot_greeks_vs_price exists in options.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QHBoxLayout, QMessageBox
)
from pricing.monte_carlo import monte_carlo_option_price
from pricing.binomial_tree import binomial_tree_option_price
from pricing.pde_solver import crank_nicolson_option_price
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox
)
from pricing.exotic_options import asian_option_price, digital_option_price
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QHBoxLayout
)
from pricing.binomial_tree_extended import one_step_binomial_call, multi_step_binomial_call
from PySide6.QtWidgets import QTabWidget, QHBoxLayout
from plotting import plot_greeks_vs_time, plot_delta_gamma_surface
from hedging_sim import simulate_delta_hedging, simulate_covered_call


class GreeksEvolutionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.s = QLineEdit()
        self.k = QLineEdit()
        self.r = QLineEdit()
        self.sigma = QLineEdit()
        self.tmax = QLineEdit()
        self.opt_type = QLineEdit()

        layout.addWidget(QLabel("Stock Price (S):"))
        layout.addWidget(self.s)
        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.k)
        layout.addWidget(QLabel("Risk-Free Rate (r):"))
        layout.addWidget(self.r)
        layout.addWidget(QLabel("Volatility (σ):"))
        layout.addWidget(self.sigma)
        layout.addWidget(QLabel("Max Time to Maturity (T):"))
        layout.addWidget(self.tmax)
        layout.addWidget(QLabel("Option Type (call/put):"))
        layout.addWidget(self.opt_type)

        btn = QPushButton("Plot Greeks vs Time")
        btn.clicked.connect(self.plot)
        layout.addWidget(btn)
        self.setLayout(layout)

    def plot(self):
        try:
            plot_greeks_vs_time(
                float(self.s.text()),
                float(self.k.text()),
                float(self.r.text()),
                float(self.sigma.text()),
                self.opt_type.text().strip().lower(),
                float(self.tmax.text())
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class SurfacePlotTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.k = QLineEdit()
        self.r = QLineEdit()
        self.sigma = QLineEdit()
        self.opt_type = QLineEdit()

        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.k)
        layout.addWidget(QLabel("Risk-Free Rate (r):"))
        layout.addWidget(self.r)
        layout.addWidget(QLabel("Volatility (σ):"))
        layout.addWidget(self.sigma)
        layout.addWidget(QLabel("Option Type (call/put):"))
        layout.addWidget(self.opt_type)

        btn = QPushButton("Plot Delta & Gamma Surface")
        btn.clicked.connect(self.plot)
        layout.addWidget(btn)
        self.setLayout(layout)

    def plot(self):
        try:
            plot_delta_gamma_surface(
                float(self.k.text()),
                float(self.r.text()),
                float(self.sigma.text()),
                self.opt_type.text().strip().lower()
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class HedgingSimTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.s0 = QLineEdit()
        self.k = QLineEdit()
        self.t = QLineEdit()
        self.r = QLineEdit()
        self.sigma = QLineEdit()
        self.opt_type = QLineEdit()

        layout.addWidget(QLabel("Initial Stock Price (S0):"))
        layout.addWidget(self.s0)
        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.k)
        layout.addWidget(QLabel("Time to Maturity (T):"))
        layout.addWidget(self.t)
        layout.addWidget(QLabel("Risk-Free Rate (r):"))
        layout.addWidget(self.r)
        layout.addWidget(QLabel("Volatility (σ):"))
        layout.addWidget(self.sigma)
        layout.addWidget(QLabel("Option Type (call/put):"))
        layout.addWidget(self.opt_type)

        btn1 = QPushButton("Simulate Delta Hedging")
        btn2 = QPushButton("Simulate Covered Call")

        btn1.clicked.connect(self.run_delta_hedge)
        btn2.clicked.connect(self.run_covered_call)

        layout.addWidget(btn1)
        layout.addWidget(btn2)
        self.setLayout(layout)

    def run_delta_hedge(self):
        try:
            simulate_delta_hedging(
                float(self.s0.text()),
                float(self.k.text()),
                float(self.t.text()),
                float(self.r.text()),
                float(self.sigma.text()),
                self.opt_type.text().strip().lower()
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def run_covered_call(self):
        try:
            simulate_covered_call(
                float(self.s0.text()),
                float(self.k.text()),
                float(self.t.text()),
                float(self.r.text()),
                float(self.sigma.text())
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class BinomialPricingTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # Common inputs
        self.S_input = QLineEdit()
        self.K_input = QLineEdit()
        self.r_input = QLineEdit()
        self.method_selector = QLineEdit()
        self.method_selector.setPlaceholderText("Enter 1 for One-Step, 2 for Multi-Step")

        # One-step specific
        self.u_input = QLineEdit()
        self.d_input = QLineEdit()

        # Multi-step specific
        self.T_input = QLineEdit()
        self.sigma_input = QLineEdit()
        self.N_input = QLineEdit()

        # Labels
        layout.addWidget(QLabel("Current Stock Price (S):"))
        layout.addWidget(self.S_input)

        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.K_input)

        layout.addWidget(QLabel("Risk-free Rate (decimal):"))
        layout.addWidget(self.r_input)

        layout.addWidget(QLabel("Choose Method: 1=One-Step, 2=Multi-Step"))
        layout.addWidget(self.method_selector)

        # One-Step inputs
        layout.addWidget(QLabel("Up factor (u) [One-Step only]:"))
        layout.addWidget(self.u_input)

        layout.addWidget(QLabel("Down factor (d) [One-Step only]:"))
        layout.addWidget(self.d_input)

        # Multi-Step inputs
        layout.addWidget(QLabel("Time to Maturity (T in years) [Multi-Step only]:"))
        layout.addWidget(self.T_input)

        layout.addWidget(QLabel("Volatility (sigma) [Multi-Step only]:"))
        layout.addWidget(self.sigma_input)

        layout.addWidget(QLabel("Number of Steps (N) [Multi-Step only]:"))
        layout.addWidget(self.N_input)

        # Calculate button
        btn = QPushButton("Calculate Binomial Price")
        btn.clicked.connect(self.calculate)
        layout.addWidget(btn)

        self.setLayout(layout)

    def calculate(self):
        try:
            S = float(self.S_input.text())
            K = float(self.K_input.text())
            r = float(self.r_input.text())
            method = self.method_selector.text().strip()

            if method == '1':
                # One-step requires u, d
                u = float(self.u_input.text())
                d = float(self.d_input.text())
                result = one_step_binomial_call(S, K, u, d, r)
                msg = (
                    f"Risk-neutral Probability p: {result['p']:.4f}\n"
                    f"Call Value if Up (Cu): {result['Cu']:.4f}\n"
                    f"Call Value if Down (Cd): {result['Cd']:.4f}\n"
                    f"Current Call Price: {result['Call Price']:.4f}"
                )
            elif method == '2':
                T = float(self.T_input.text())
                sigma = float(self.sigma_input.text())
                N = int(self.N_input.text())
                price = multi_step_binomial_call(S, K, T, r, sigma, N)
                msg = f"Multi-Step Binomial Call Option Price: {price:.4f}"
            else:
                QMessageBox.warning(self, "Error", "Please enter method as '1' or '2'")
                return

            QMessageBox.information(self, "Result", msg)

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class ExoticOptionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.S_input = QLineEdit()
        self.K_input = QLineEdit()
        self.T_input = QLineEdit()
        self.r_input = QLineEdit()
        self.sigma_input = QLineEdit()
        self.steps_input = QLineEdit()
        self.sims_input = QLineEdit()

        self.option_type_box = QComboBox()
        self.option_type_box.addItems(["call", "put"])

        self.exotic_type_box = QComboBox()
        self.exotic_type_box.addItems(["Asian", "Digital"])

        button = QPushButton("Calculate Exotic Option Price")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Stock Price (S):"))
        layout.addWidget(self.S_input)

        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.K_input)

        layout.addWidget(QLabel("Time to Maturity (T in years):"))
        layout.addWidget(self.T_input)

        layout.addWidget(QLabel("Risk-Free Rate (r):"))
        layout.addWidget(self.r_input)

        layout.addWidget(QLabel("Volatility (sigma):"))
        layout.addWidget(self.sigma_input)

        layout.addWidget(QLabel("Number of Simulations:"))
        layout.addWidget(self.sims_input)

        layout.addWidget(QLabel("Steps per Path (only for Asian):"))
        layout.addWidget(self.steps_input)

        layout.addWidget(QLabel("Option Type:"))
        layout.addWidget(self.option_type_box)

        layout.addWidget(QLabel("Exotic Option Type:"))
        layout.addWidget(self.exotic_type_box)

        layout.addWidget(button)
        self.setLayout(layout)

    def calculate(self):
        try:
            S = float(self.S_input.text())
            K = float(self.K_input.text())
            T = float(self.T_input.text())
            r = float(self.r_input.text())
            sigma = float(self.sigma_input.text())
            sims = int(self.sims_input.text())
            option_type = self.option_type_box.currentText()
            exotic_type = self.exotic_type_box.currentText()

            if exotic_type == "Asian":
                steps = int(self.steps_input.text())
                price = asian_option_price(S, K, T, r, sigma, option_type, sims, steps)
            elif exotic_type == "Digital":
                price = digital_option_price(S, K, T, r, sigma, option_type, sims)
            else:
                raise ValueError("Invalid exotic option type.")

            QMessageBox.information(self, "Result", f"{exotic_type} {option_type.capitalize()} Option Price: ${price:.4f}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class AdvancedPricingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.S_input = QLineEdit()
        self.K_input = QLineEdit()
        self.T_input = QLineEdit()
        self.r_input = QLineEdit()
        self.sigma_input = QLineEdit()

        self.option_type_box = QComboBox()
        self.option_type_box.addItems(["call", "put"])

        self.method_box = QComboBox()
        self.method_box.addItems(["Monte Carlo", "Binomial Tree", "Crank-Nicolson PDE"])

        self.steps_input = QLineEdit()  # Used for simulations or tree steps
        self.steps_input.setPlaceholderText("Steps/Simulations (e.g., 100000)")

        self.american_checkbox = QComboBox()
        self.american_checkbox.addItems(["No", "Yes"])

        button = QPushButton("Calculate")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Stock Price (S):"))
        layout.addWidget(self.S_input)

        layout.addWidget(QLabel("Strike Price (K):"))
        layout.addWidget(self.K_input)

        layout.addWidget(QLabel("Time to Maturity (T in years):"))
        layout.addWidget(self.T_input)

        layout.addWidget(QLabel("Risk-Free Rate (r):"))
        layout.addWidget(self.r_input)

        layout.addWidget(QLabel("Volatility (sigma):"))
        layout.addWidget(self.sigma_input)

        layout.addWidget(QLabel("Option Type:"))
        layout.addWidget(self.option_type_box)

        layout.addWidget(QLabel("Pricing Method:"))
        layout.addWidget(self.method_box)

        layout.addWidget(QLabel("Steps/Simulations:"))
        layout.addWidget(self.steps_input)

        layout.addWidget(QLabel("Is American Option? (only for Binomial):"))
        layout.addWidget(self.american_checkbox)

        layout.addWidget(button)
        self.setLayout(layout)

    def calculate(self):
        try:
            S = float(self.S_input.text())
            K = float(self.K_input.text())
            T = float(self.T_input.text())
            r = float(self.r_input.text())
            sigma = float(self.sigma_input.text())
            option_type = self.option_type_box.currentText()
            method = self.method_box.currentText()
            steps = int(self.steps_input.text()) if self.steps_input.text() else 100
            is_american = self.american_checkbox.currentText() == "Yes"

            if method == "Monte Carlo":
                price = monte_carlo_option_price(S, K, T, r, sigma, option_type, steps)
            elif method == "Binomial Tree":
                price = binomial_tree_option_price(S, K, T, r, sigma, steps, option_type, is_american)
            elif method == "Crank-Nicolson PDE":
                price = crank_nicolson_option_price(S, K, T, r, sigma, option_type=option_type)
            else:
                raise ValueError("Unknown method")

            QMessageBox.information(self, "Result", f"{method} {option_type.capitalize()} Option Price: ${price:.4f}")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class SimpleInterestTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_p = QLineEdit()
        self.input_r = QLineEdit()
        self.input_t = QLineEdit()
        button = QPushButton("Calculate Simple Interest")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Principal (P):"))
        layout.addWidget(self.input_p)
        layout.addWidget(QLabel("Annual interest rate (decimal):"))
        layout.addWidget(self.input_r)
        layout.addWidget(QLabel("Time in years (t):"))
        layout.addWidget(self.input_t)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            P = float(self.input_p.text())
            r = float(self.input_r.text())
            t = float(self.input_t.text())
            si = SimpleInterest(P, r, t)
            QMessageBox.information(self, "Simple Interest", f"Simple Interest: {si.calculate():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class CompoundInterestTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_p = QLineEdit()
        self.input_r = QLineEdit()
        self.input_t = QLineEdit()
        self.input_n = QLineEdit()
        button = QPushButton("Calculate Compound Interest")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Principal (P):"))
        layout.addWidget(self.input_p)
        layout.addWidget(QLabel("Annual interest rate (decimal):"))
        layout.addWidget(self.input_r)
        layout.addWidget(QLabel("Time in years (t):"))
        layout.addWidget(self.input_t)
        layout.addWidget(QLabel("Times compounded per year (n):"))
        layout.addWidget(self.input_n)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            P = float(self.input_p.text())
            r = float(self.input_r.text())
            t = float(self.input_t.text())
            n = int(self.input_n.text())
            ci = CompoundInterest(P, r, t, n)
            QMessageBox.information(self, "Compound Interest", f"Compound Interest: {ci.calculate():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class ContinuousCompoundingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_p = QLineEdit()
        self.input_r = QLineEdit()
        self.input_t = QLineEdit()
        button = QPushButton("Calculate Continuous Compounding")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Principal (P):"))
        layout.addWidget(self.input_p)
        layout.addWidget(QLabel("Annual interest rate (decimal):"))
        layout.addWidget(self.input_r)
        layout.addWidget(QLabel("Time in years (t):"))
        layout.addWidget(self.input_t)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            P = float(self.input_p.text())
            r = float(self.input_r.text())
            t = float(self.input_t.text())
            cc = ContinuousCompounding(P, r, t)
            QMessageBox.information(self, "Continuous Compounding", f"Interest: {cc.calculate():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class ZeroCouponBondTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_F = QLineEdit()
        self.input_P = QLineEdit()
        self.input_t = QLineEdit()
        button = QPushButton("Calculate Zero-Coupon Bond YTM")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Face value (F):"))
        layout.addWidget(self.input_F)
        layout.addWidget(QLabel("Price (P):"))
        layout.addWidget(self.input_P)
        layout.addWidget(QLabel("Time to maturity (years):"))
        layout.addWidget(self.input_t)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            F = float(self.input_F.text())
            P = float(self.input_P.text())
            t = float(self.input_t.text())
            bond = ZeroCouponBond(F, P, t)
            ytm = bond.ytm()
            QMessageBox.information(self, "Zero-Coupon Bond YTM", f"YTM: {ytm * 100:.4f}%")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class CouponBearingBondTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_F = QLineEdit()
        self.input_c = QLineEdit()
        self.input_P = QLineEdit()
        self.input_t = QLineEdit()
        self.input_freq = QLineEdit()

        self.ytm_type_combo = QComboBox()
        self.ytm_type_combo.addItems([
            "Simplified YTM",
            "Original YTM",
            "Continuous Compounding YTM"
        ])

        button = QPushButton("Calculate YTM")
        button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Face value (F):"))
        layout.addWidget(self.input_F)
        layout.addWidget(QLabel("Annual coupon rate (decimal):"))
        layout.addWidget(self.input_c)
        layout.addWidget(QLabel("Price (P):"))
        layout.addWidget(self.input_P)
        layout.addWidget(QLabel("Time to maturity (years):"))
        layout.addWidget(self.input_t)
        layout.addWidget(QLabel("Coupon payments per year:"))
        layout.addWidget(self.input_freq)
        layout.addWidget(QLabel("YTM Type:"))
        layout.addWidget(self.ytm_type_combo)
        layout.addWidget(button)

        self.setLayout(layout)

    def calculate(self):
        try:
            F = float(self.input_F.text())
            c = float(self.input_c.text())
            P = float(self.input_P.text())
            t = float(self.input_t.text())
            freq = int(self.input_freq.text())
            bond = CouponBearingBond(F, c, P, t, freq)
            ytm_type = self.ytm_type_combo.currentText()

            if ytm_type == "Simplified YTM":
                ytm = bond.ytm_simplified()
            elif ytm_type == "Original YTM":
                ytm = bond.ytm_original()
            else:
                ytm = bond.ytm_continuous()

            QMessageBox.information(self, "Coupon-Bearing Bond YTM", f"{ytm_type}: {ytm * 100:.4f}%")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class OptionPricingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        grid = QGridLayout()

        self.inputs = {}
        labels = [
            "Stock Price (S):",
            "Strike Price (K):",
            "Time to Maturity (years):",
            "Risk-Free Rate (r):",
            "Volatility (sigma):"
        ]

        for i, label in enumerate(labels):
            grid.addWidget(QLabel(label), i, 0)
            line_edit = QLineEdit()
            grid.addWidget(line_edit, i, 1)
            self.inputs[label] = line_edit

        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        grid.addWidget(QLabel("Option Type:"), len(labels), 0)
        grid.addWidget(self.option_type_combo, len(labels), 1)

        layout.addLayout(grid)

        self.calc_button = QPushButton("Calculate Option Price & Greeks")
        self.calc_button.clicked.connect(self.calculate_option)
        layout.addWidget(self.calc_button)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def calculate_option(self):
        try:
            S = float(self.inputs["Stock Price (S):"].text())
            K = float(self.inputs["Strike Price (K):"].text())
            T = float(self.inputs["Time to Maturity (years):"].text())
            r = float(self.inputs["Risk-Free Rate (r):"].text())
            sigma = float(self.inputs["Volatility (sigma):"].text())
            option_type = self.option_type_combo.currentText()

            opt = Option(S, K, T, r, sigma, option_type)
            price = opt.price()
            greeks = opt.greeks()
            result = f"<b>{option_type.capitalize()} Option Price:</b> ${price:.4f}<br><b>Greeks:</b><br>"
            for k, v in greeks.items():
                result += f"{k}: {v:.6f}<br>"
            self.result_label.setText(result)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class OptionPlotTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        grid = QGridLayout()
        self.inputs = {}
        labels = [
            "Strike Price (K):",
            "Time to Maturity (years):",
            "Risk-Free Rate (r):",
            "Volatility (sigma):",
            "Min Stock Price for Plot:",
            "Max Stock Price for Plot:",
            "Step Size for Plot:"
        ]
        for i, label in enumerate(labels):
            grid.addWidget(QLabel(label), i, 0)
            line_edit = QLineEdit()
            grid.addWidget(line_edit, i, 1)
            self.inputs[label] = line_edit

        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        grid.addWidget(QLabel("Option Type:"), len(labels), 0)
        grid.addWidget(self.option_type_combo, len(labels), 1)

        layout.addLayout(grid)

        btn = QPushButton("Plot Greeks vs Stock Price")
        btn.clicked.connect(self.plot_greeks)
        layout.addWidget(btn)

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot_greeks(self):
        try:
            K = float(self.inputs["Strike Price (K):"].text())
            T = float(self.inputs["Time to Maturity (years):"].text())
            r = float(self.inputs["Risk-Free Rate (r):"].text())
            sigma = float(self.inputs["Volatility (sigma):"].text())
            s_min = float(self.inputs["Min Stock Price for Plot:"].text())
            s_max = float(self.inputs["Max Stock Price for Plot:"].text())
            step = float(self.inputs["Step Size for Plot:"].text())
            option_type = self.option_type_combo.currentText()

            S_vals = np.arange(s_min, s_max + step, step)

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

            self.figure.clear()
            axs = self.figure.subplots(3, 2)
            self.figure.suptitle(f"{option_type.capitalize()} Option Greeks vs Stock Price", fontsize=16)

            axs[0, 0].plot(S_vals, price_vals, 'b')
            axs[0, 0].set_title("Price")
            axs[0, 0].set_xlabel("Stock Price")
            axs[0, 0].set_ylabel("Option Price")

            axs[0, 1].plot(S_vals, delta_vals, 'r')
            axs[0, 1].set_title("Delta")
            axs[0, 1].set_xlabel("Stock Price")
            axs[0, 1].set_ylabel("Delta")

            axs[1, 0].plot(S_vals, gamma_vals, 'g')
            axs[1, 0].set_title("Gamma")
            axs[1, 0].set_xlabel("Stock Price")
            axs[1, 0].set_ylabel("Gamma")

            axs[1, 1].plot(S_vals, theta_vals, 'm')
            axs[1, 1].set_title("Theta")
            axs[1, 1].set_xlabel("Stock Price")
            axs[1, 1].set_ylabel("Theta")

            axs[2, 0].plot(S_vals, vega_vals, 'c')
            axs[2, 0].set_title("Vega")
            axs[2, 0].set_xlabel("Stock Price")
            axs[2, 0].set_ylabel("Vega")

            axs[2, 1].plot(S_vals, rho_vals, 'y')
            axs[2, 1].set_title("Rho")
            axs[2, 1].set_xlabel("Stock Price")
            axs[2, 1].set_ylabel("Rho")

            self.figure.tight_layout(rect=[0, 0, 1, 0.96])
            self.canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


class FinanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finance Calculator")

        tabs = QTabWidget()
        tabs.addTab(SimpleInterestTab(), "Simple Interest")
        tabs.addTab(CompoundInterestTab(), "Compound Interest")
        tabs.addTab(ContinuousCompoundingTab(), "Continuous Compounding")
        tabs.addTab(ZeroCouponBondTab(), "Zero-Coupon Bond")
        tabs.addTab(CouponBearingBondTab(), "Coupon-Bearing Bond")
        tabs.addTab(OptionPricingTab(), "Option Pricing")
        tabs.addTab(OptionPlotTab(), "Option Greeks Plot")
        tabs.addTab(AdvancedPricingTab(), "Advanced Pricing")
        tabs.addTab(ExoticOptionTab(), "Exotic Options")
        tabs.addTab(BinomialPricingTab(), "Binomial Pricing")
        tabs.addTab(SimpleInterestTab(), "Simple Interest")
        tabs.addTab(GreeksEvolutionTab(), "Greeks vs Time")
        tabs.addTab(SurfacePlotTab(), "Delta/Gamma Surfaces")
        tabs.addTab(HedgingSimTab(), "Hedging Simulation")

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinanceApp()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
