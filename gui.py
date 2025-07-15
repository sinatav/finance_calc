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

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinanceApp()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
