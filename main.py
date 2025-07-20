import numpy as np
import ast
from interests import SimpleInterest, CompoundInterest, ContinuousCompounding
from bonds import ZeroCouponBond, CouponBearingBond
from options import Option, plot_greeks_vs_price
from pricing.monte_carlo import monte_carlo_option_price
from pricing.binomial_tree import binomial_tree_option_price
from pricing.pde_solver import crank_nicolson_option_price
from pricing.exotic_options import asian_option_price, digital_option_price
from pricing.binomial_tree_extended import one_step_binomial_call, multi_step_binomial_call
from plotting import plot_greeks_vs_time, plot_delta_gamma_surface
from hedging_sim import simulate_delta_hedging, simulate_covered_call
from risk.var_cvar import compute_var_cvar, visualize_var_cvar
from risk.stress_test import StressTestSimulator
from portfolio.expected_return import expected_portfolio_return
from portfolio.portfolio_variance import portfolio_variance
from portfolio.capital_market_line import capital_market_line
from portfolio.capm import capital_asset_pricing_model
from portfolio.beta import calculate_beta
from portfolio.sharpe import sharpe_ratio
from portfolio.efficient_frontier import plot_efficient_frontier
from risk.var import parametric_var
from risk.cvar import conditional_var
from risk.scaling_var import scale_var
from time_series.moving_average import moving_average, exponential_moving_average
from time_series.ar_model import autoregressive_model
from time_series.arima_model import fit_arima
from models.scaling_var import scale_var
from models.moving_average import calculate_sma, calculate_ema
from models.ar_model import fit_ar_model
from models.ma_model import fit_ma_model
from models.arima_model import fit_arima_model
from models_menu import ui
from forecasting.kalman_filter import run_kalman_filter_
from forecasting.random_forest import run_random_forest_model
from forecasting.neural_network import run_neural_network_model
from forecasting.garch_model import run_garch_forecasting
from models.least_squares import least_squares_regression
from models.rsquared import r_squared
from models.nonlinear_regression import polynomial_fit
from models.time_value_money import (
    future_value,
    present_value,
    annuity_value,
    perpetuity_value
)



def main_menu():
    print("\nFinance Calculator Menu:")
    print("1. Simple Interest")
    print("2. Compound Interest")
    print("3. Continuous Compounding")
    print("4. Zero-Coupon Bond YTM")
    print("5. Coupon-Bearing Bond YTM (Simplified)")
    print("6. Coupon-Bearing Bond YTM (Original)")
    print("7. Coupon-Bearing Bond YTM (Continuous Compounding)")
    print("8. Black-Scholes Option Pricing & Greeks")
    print("9. Plot Option Greeks vs Stock Price")
    print("10. Monte Carlo Option Pricing")
    print("11. Binomial Tree Option Pricing")
    print("12. Crank-Nicolson PDE Option Pricing")
    print("13. Asian Option Pricing (Monte Carlo)")
    print("14. Digital Option Pricing (Monte Carlo)")
    print("15. One-Step Binomial Call Option")
    print("16. Multi-Step Binomial Call Option")
    print("17. Hedging Calculations (Delta, Gamma, Theta, Vega)")
    print("18. Plot Greeks Evolution Over Time")
    print("19. Plot Delta/Gamma Surfaces")
    print("20. Simulate Delta Hedging Strategy")
    print("21. Simulate Covered Call Strategy")
    print("22. Risk Analysis (VaR / CVaR)")
    print("23. Run Stress Test Simulation")
    print("24. Calculate Expected Portfolio Return")
    print("25. Calculate Portfolio Variance")
    print("26. Calculate Capital Market Line (CML) Return")
    print("27. Calculate Expected Return using CAPM")
    print("28. Calculate Beta of an asset")
    print("29. Calculate Sharpe Ratio")
    print("30. Plot Efficient Frontier")
    print("31. Calculate Basic Parametric VaR")
    print("32. Calculate Conditional VaR (CVaR)")
    print("33. Scale VaR for Multiple Days")
    print("34. Calculate Moving Average (MA)")
    print("35. Calculate Exponential Moving Average (EMA)")
    print("36. Autoregressive (AR) Model Prediction")
    print("37. Fit ARIMA Model")
    print("38. Enter Models' Menu")
    print("39. Kalman Filter (Recursive Estimation)")
    print("40. Random Forest Model (Machine Learning)")
    print("41. Neural Network Model (Machine Learning)")
    print("42. GARCH Model (Volatility Forecasting)")
    print("43. Least Squares Regression")
    print("44. R-squared (Goodness of Fit)")
    print("45. Nonlinear & Polynomial Regression")
    print("46. Time Value of Money")
    print("0. Exit")


def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def run_simple_interest():
    P = get_float("Principal (P): ")
    r = get_float("Annual interest rate (as decimal, e.g., 0.05): ")
    t = get_float("Time in years (t): ")
    si = SimpleInterest(P, r, t)
    interest = si.calculate()
    print(f"Simple Interest: {interest:.2f}")


def run_compound_interest():
    P = get_float("Principal (P): ")
    r = get_float("Annual interest rate (as decimal, e.g., 0.05): ")
    t = get_float("Time in years (t): ")
    n = int(get_float("Number of times interest compounded per year (n): "))
    ci = CompoundInterest(P, r, t, n)
    interest = ci.calculate()
    print(f"Compound Interest: {interest:.2f}")


def run_continuous_compounding():
    P = get_float("Principal (P): ")
    r = get_float("Annual interest rate (as decimal, e.g., 0.05): ")
    t = get_float("Time in years (t): ")
    cc = ContinuousCompounding(P, r, t)
    interest = cc.calculate()
    print(f"Interest with Continuous Compounding: {interest:.2f}")


def run_zero_coupon_bond():
    F = get_float("Face value (F): ")
    P = get_float("Price (P): ")
    t = get_float("Time to maturity in years (t): ")
    bond = ZeroCouponBond(F, P, t)
    ytm = bond.ytm()
    print(f"Zero-Coupon Bond Yield to Maturity (YTM): {ytm * 100:.4f}%")


def run_coupon_bearing_bond_simplified():
    F = get_float("Face value (F): ")
    c = get_float("Annual coupon rate (decimal, e.g., 0.06): ")
    P = get_float("Price (P): ")
    t = get_float("Time to maturity in years (t): ")
    freq = int(get_float("Coupon payments per year (e.g., 2 for semiannual): "))
    bond = CouponBearingBond(F, c, P, t, freq)
    ytm = bond.ytm_simplified()
    print(f"Simplified Coupon-Bearing Bond YTM: {ytm * 100:.4f}%")


def run_coupon_bearing_bond_original():
    F = get_float("Face value (F): ")
    c = get_float("Annual coupon rate (decimal, e.g., 0.06): ")
    P = get_float("Price (P): ")
    t = get_float("Time to maturity in years (t): ")
    freq = int(get_float("Coupon payments per year (e.g., 2 for semiannual): "))
    bond = CouponBearingBond(F, c, P, t, freq)
    ytm = bond.ytm_original()
    print(f"Original Coupon-Bearing Bond YTM: {ytm * 100:.4f}%")


def run_coupon_bearing_bond_continuous():
    F = get_float("Face value (F): ")
    c = get_float("Annual coupon rate (decimal, e.g., 0.06): ")
    P = get_float("Price (P): ")
    t = get_float("Time to maturity in years (t): ")
    freq = int(get_float("Coupon payments per year (e.g., 2 for semiannual): "))
    bond = CouponBearingBond(F, c, P, t, freq)
    ytm = bond.ytm_continuous()
    print(f"Continuous Compounding Coupon-Bearing Bond YTM: {ytm*100:.4f}%")


def run_option_pricing():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (in years): ")
    r = get_float("Risk-free rate (as decimal): ")
    sigma = get_float("Volatility (as decimal, e.g., 0.2 for 20%): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()

    try:
        opt = Option(S, K, T, r, sigma, option_type)
        price = opt.price()
        greeks = opt.greeks()
        print(f"\n{option_type.capitalize()} Option Price: ${price:.2f}")
        print("Option Greeks:")
        for greek, val in greeks.items():
            print(f"  {greek}: {val:.4f}")
    except ValueError as e:
        print(f"Error: {e}")


def run_option_plot():
    print("Enter parameters for plotting Option Greeks:")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal, e.g., 0.2): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    s_min = get_float("Minimum stock price to plot: ")
    s_max = get_float("Maximum stock price to plot: ")
    step = get_float("Step size: ")

    plot_greeks_vs_price(K, T, r, sigma, option_type, S_range=(s_min, s_max), step=step)


def run_monte_carlo():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    sims = int(get_float("Number of simulations (e.g., 100000): "))

    price = monte_carlo_option_price(S, K, T, r, sigma, option_type, sims)
    print(f"Monte Carlo Estimated {option_type.capitalize()} Option Price: ${price:.4f}")


def run_binomial_tree():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    steps = int(get_float("Number of steps (e.g., 100): "))
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    is_american = input("Is it an American option? (y/n): ").strip().lower() == 'y'

    price = binomial_tree_option_price(S, K, T, r, sigma, steps, option_type, is_american)
    print(f"Binomial Tree {option_type.capitalize()} Option Price: ${price:.4f}")


def run_crank_nicolson():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()

    price = crank_nicolson_option_price(S, K, T, r, sigma, option_type=option_type)
    print(f"Crank-Nicolson {option_type.capitalize()} Option Price: ${price:.4f}")


def run_asian_option():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    steps = int(get_float("Steps per path (e.g. 100): "))
    sims = int(get_float("Number of simulations (e.g. 100000): "))
    option_type = input("Option type ('call' or 'put'): ").strip().lower()

    price = asian_option_price(S, K, T, r, sigma, option_type, sims, steps)
    print(f"Asian {option_type.capitalize()} Option Price: ${price:.4f}")


def run_digital_option():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    sims = int(get_float("Number of simulations (e.g. 100000): "))
    option_type = input("Option type ('call' or 'put'): ").strip().lower()

    price = digital_option_price(S, K, T, r, sigma, option_type, sims)
    print(f"Digital {option_type.capitalize()} Option Price: ${price:.4f}")

def run_one_step_binomial():
    S = get_float("Current stock price (S): ")
    K = get_float("Strike price (K): ")
    u = get_float("Up factor (u): ")
    d = get_float("Down factor (d): ")
    r = get_float("Risk-free rate (decimal): ")

    result = one_step_binomial_call(S, K, u, d, r)
    print(f"Risk-neutral probability (p): {result['p']:.4f}")
    print(f"Call Value if Up (Cu): {result['Cu']:.4f}")
    print(f"Call Value if Down (Cd): {result['Cd']:.4f}")
    print(f"Current Call Price: {result['Call Price']:.4f}")


def run_multi_step_binomial():
    S = get_float("Current stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (in years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (sigma): ")
    N = int(get_float("Number of steps (N): "))

    price = multi_step_binomial_call(S, K, T, r, sigma, N)
    print(f"Multi-Step Binomial Call Option Price: {price:.4f}")


def run_hedging():
    print("Base Option Parameters:")
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    base_opt = Option(S, K, T, r, sigma, option_type)

    print("\nHedge Option Parameters:")
    S_h = get_float("Stock price (S): ")
    K_h = get_float("Strike price (K): ")
    T_h = get_float("Time to maturity (years): ")
    r_h = get_float("Risk-free rate (decimal): ")
    sigma_h = get_float("Volatility (decimal): ")
    option_type_h = input("Option type ('call' or 'put'): ").strip().lower()
    hedge_opt = Option(S_h, K_h, T_h, r_h, sigma_h, option_type_h)

    target_exp = int(get_float("Number of base option contracts to hedge: "))

    shares = base_opt.delta_hedge_shares(target_exp)
    gamma_contracts = base_opt.gamma_hedge_options(hedge_opt, target_exp)
    theta_vega = base_opt.theta_vega_positioning(hedge_opt, target_exp)

    print(f"\nDelta hedge: {shares:.4f} shares of underlying")
    print(f"Gamma hedge: {gamma_contracts:.4f} contracts of hedge option")
    print(f"Theta hedge: {theta_vega['theta_hedge']} contracts of hedge option")
    print(f"Vega hedge: {theta_vega['vega_hedge']} contracts of hedge option")


def run_plot_greeks_time():
    S = get_float("Stock price (S): ")
    K = get_float("Strike price (K): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    T_max = get_float("Max time to maturity (years, e.g., 1): ")
    plot_greeks_vs_time(S, K, r, sigma, option_type, T_max)


def run_plot_greeks_surface():
    K = get_float("Strike price (K): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    plot_delta_gamma_surface(K, r, sigma, option_type)


def run_simulate_delta_hedging():
    S0 = get_float("Initial stock price (S0): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    simulate_delta_hedging(S0, K, T, r, sigma, option_type)


def run_simulate_covered_call():
    S0 = get_float("Initial stock price (S0): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to maturity (years): ")
    r = get_float("Risk-free rate (decimal): ")
    sigma = get_float("Volatility (decimal): ")
    simulate_covered_call(S0, K, T, r, sigma)


def risk_analysis_menu():
    returns = np.random.normal(0.001, 0.02, 1000)  # Simulated return series
    confidence = float(input("Enter confidence level (e.g. 0.95): ") or 0.95)
    var, cvar = compute_var_cvar(returns, confidence)
    print(f"Value at Risk (VaR): {var:.4f}")
    print(f"Conditional Value at Risk (CVaR): {cvar:.4f}")
    visualize = input("Visualize histogram and VaR/CVaR? (y/n): ").lower() == 'y'
    if visualize:
        visualize_var_cvar(returns, confidence)


def run_stress_test_simulation():
    returns = np.random.normal(0.001, 0.02, 1000)
    simulator = StressTestSimulator(returns)
    shocks = {
        "Mild": -0.05,
        "Moderate": -0.15,
        "Severe": -0.3
    }
    result = simulator.run_stress_scenario(shocks)
    print("\nStress Test Summary:\n", result)
    simulator.plot_shock_effects(shocks)


def run_expected_return_simple():
    print("Enter the weights of assets (comma-separated, e.g., 0.5,0.3,0.2):")
    weights = list(map(float, input().split(',')))
    print("Enter the expected returns for each asset (comma-separated, e.g., 0.08,0.05,0.12):")
    expected_returns = list(map(float, input().split(',')))
    result = expected_portfolio_return(weights, expected_returns)
    print(f"Expected Portfolio Return: {result:.4f}")


def calc_portfolio_variance():
    print("Enter weights (comma-separated, e.g., 0.5,0.5):")
    weights = list(map(float, input().split(',')))
    print("Enter the covariance matrix (e.g., for 2 assets: 0.01,0.002;0.002,0.02):")
    rows = input().split(';')
    covariance_matrix = [list(map(float, row.split(','))) for row in rows]
    result = portfolio_variance(weights, covariance_matrix)
    print(f"Portfolio Variance: {result:.6f}")


def calc_cml():
    rf = float(input("Enter risk-free rate (Rf): "))
    rm = float(input("Enter expected market return (Rm): "))
    sigma_m = float(input("Enter market standard deviation (σm): "))
    sigma_c = float(input("Enter your portfolio standard deviation (σc): "))
    result = capital_market_line(rf, rm, sigma_m, sigma_c)
    print(f"Expected Return on CML: {result:.6f}")


def calc_capm():
    rf = float(input("Enter risk-free rate (Rf): "))
    beta = float(input("Enter beta of the asset (β): "))
    rm = float(input("Enter expected market return (Rm): "))
    result = capital_asset_pricing_model(rf, beta, rm)
    print(f"Expected Return using CAPM: {result:.6f}")


def calc_beta():
    asset_str = input("Enter asset returns as list (e.g., [0.01, 0.02, -0.005]): ")
    market_str = input("Enter market returns as list (e.g., [0.015, 0.025, -0.003]): ")
    asset_returns = ast.literal_eval(asset_str)
    market_returns = ast.literal_eval(market_str)
    beta = calculate_beta(asset_returns, market_returns)
    print(f"Beta (β) of the asset: {beta:.6f}")


def cli_sharpe_ratio():
    print("\n--- Sharpe Ratio Calculation ---")
    expected_return = float(input("Enter expected portfolio return (decimal, e.g. 0.1 for 10%): "))
    risk_free_rate = float(input("Enter risk-free rate (decimal, e.g. 0.02 for 2%): "))
    portfolio_std_dev = float(input("Enter portfolio standard deviation (risk): "))

    sharpe = sharpe_ratio(expected_return, risk_free_rate, portfolio_std_dev)
    print(f"Sharpe Ratio: {sharpe:.4f}")


def cli_efficient_frontier():
    print("\n--- Efficient Frontier Plot ---")
    n = int(input("Enter number of assets in portfolio: "))
    returns = []
    print("Enter expected returns of each asset (decimal):")
    for i in range(n):
        r = float(input(f"Asset {i+1}: "))
        returns.append(r)

    print("Enter covariance matrix (each row as space-separated decimals):")
    cov_matrix = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != n:
            print("Invalid row length. Aborting.")
            return
        cov_matrix.append(row)

    plot_efficient_frontier(returns, cov_matrix)
    print("Plot generated successfully.")


def cli_parametric_var():
    print("\n--- Basic Parametric VaR ---")
    returns = list(map(float, input("Enter portfolio returns as space-separated decimals: ").split()))
    confidence_level = float(input("Enter confidence level (e.g., 0.95): "))

    var = parametric_var(returns, confidence_level)
    print(f"Parametric VaR at {confidence_level*100:.2f}% confidence: {var:.4f}")


def cli_conditional_var():
    print("\n--- Conditional VaR (CVaR) ---")
    returns = list(map(float, input("Enter portfolio returns as space-separated decimals: ").split()))
    confidence_level = float(input("Enter confidence level (e.g., 0.95): "))

    cvar = conditional_var(returns, confidence_level)
    print(f"Conditional VaR (Expected Shortfall) at {confidence_level*100:.2f}% confidence: {cvar:.4f}")


def cli_scale_var():
    print("\n--- VaR Scaling ---")
    var_1day = float(input("Enter 1-day VaR: "))
    days = int(input("Enter number of days to scale to: "))
    var_t = scale_var(var_1day, days)
    print(f"Scaled VaR for {days} days: {var_t:.4f}")


def cli_moving_average():
    print("\n--- Moving Average (MA) ---")
    data = list(map(float, input("Enter data points separated by space: ").split()))
    window = int(input("Enter window size: "))
    ma = moving_average(data, window)
    print(f"Moving Average (last values): {ma[-5:]}")  # show last 5 for brevity


def cli_exponential_moving_average():
    print("\n--- Exponential Moving Average (EMA) ---")
    data = list(map(float, input("Enter data points separated by space: ").split()))
    alpha = float(input("Enter smoothing factor alpha (0 < alpha < 1): "))
    ema = exponential_moving_average(data, alpha)
    print(f"EMA (last values): {ema[-5:]}")


def cli_ar_model():
    print("\n--- Autoregressive (AR) Model ---")
    data = list(map(float, input("Enter data points separated by space: ").split()))
    p = int(input("Enter order p of AR model: "))
    coeffs = []
    for i in range(p):
        c = float(input(f"Enter AR coefficient phi_{i+1}: "))
        coeffs.append(c)
    c0 = float(input("Enter constant term c: "))
    pred = autoregressive_model(data, coeffs, c0)
    print(f"AR Model Predictions (last 5): {pred[-5:]}")


def cli_arima_model():
    print("\n--- ARIMA Model Fit ---")
    data = list(map(float, input("Enter data points separated by space: ").split()))
    p = int(input("Enter AR order p: "))
    d = int(input("Enter degree of differencing d: "))
    q = int(input("Enter MA order q: "))
    model_fit = fit_arima(data, (p, d, q))
    print(model_fit.summary())


def run_kalman_filter():
    print("Kalman Filter (Recursive Estimation)")
    try:
        data_path = input("Enter path to time series CSV (with a 'value' column): ")
        run_kalman_filter(data_path)
    except Exception as e:
        print(f"[Error] Kalman Filter failed: {e}")

def run_random_forest():
    print("Random Forest (Classification/Regression)")
    try:
        data_path = input("Enter path to CSV file: ")
        target_column = input("Enter target column name: ")
        run_random_forest_model(data_path, target_column)
    except Exception as e:
        print(f"[Error] Random Forest failed: {e}")

def run_neural_network():
    print("Neural Network (MLP)")
    try:
        data_path = input("Enter path to CSV file: ")
        target_column = input("Enter target column name: ")
        run_neural_network_model(data_path, target_column)
    except Exception as e:
        print(f"[Error] Neural Network failed: {e}")

def run_garch_model():
    print("GARCH Model (Volatility Forecasting)")
    try:
        data_path = input("Enter path to time series CSV (with a 'returns' column): ")
        run_garch_forecasting(data_path)
    except Exception as e:
        print(f"[Error] GARCH Model failed: {e}")


def run_least_squares():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    least_squares_regression(x, y)


def run_r_squared():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([0.8, 2.1, 2.9, 4.2])
    r_squared(y_true, y_pred)


def run_polynomial_regression():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1.2, 1.9, 3.0, 4.1, 5.3])
    polynomial_fit(x, y, degree=2)


def run_time_value_money():
    print("TVM Options:\n1. Future Value\n2. Present Value\n3. Annuity\n4. Perpetuity")
    sub = input("Enter choice: ")
    if sub == "1":
        print("FV =", future_value(1000, 0.05, 5))
    elif sub == "2":
        print("PV =", present_value(1276.28, 0.05, 5))
    elif sub == "3":
        print("Annuity =", annuity_value(100, 0.05, 10))
    elif sub == "4":
        print("Perpetuity =", perpetuity_value(100, 0.05))
    else:
        print("Invalid sub-choice.")


def main():
    while True:
        main_menu()
        choice = input("Choose an option: ")
        if choice == '1':
            run_simple_interest()
        elif choice == '2':
            run_compound_interest()
        elif choice == '3':
            run_continuous_compounding()
        elif choice == '4':
            run_zero_coupon_bond()
        elif choice == '5':
            run_coupon_bearing_bond_simplified()
        elif choice == '6':
            run_coupon_bearing_bond_original()
        elif choice == '7':
            run_coupon_bearing_bond_continuous()
        elif choice == '8':
            run_option_pricing()
        elif choice == '9':
            run_option_plot()
        elif choice == '10':
            run_monte_carlo()
        elif choice == '11':
            run_binomial_tree()
        elif choice == '12':
            run_crank_nicolson()
        elif choice == '13':
            run_asian_option()
        elif choice == '14':
            run_digital_option()
        elif choice == '15':
            run_one_step_binomial()
        elif choice == '16':
            run_multi_step_binomial()
        elif choice == '17':
            run_hedging()
        elif choice == '18':
            run_plot_greeks_time()
        elif choice == '19':
            run_plot_greeks_surface()
        elif choice == '20':
            run_simulate_delta_hedging()
        elif choice == '21':
            run_simulate_covered_call()
        elif choice == "22":
            risk_analysis_menu()
        elif choice == "23":
            run_stress_test_simulation()
        elif choice == "24":
            run_expected_return_simple()
        elif choice == "25":
            calc_portfolio_variance()
        elif choice == "26":
            calc_cml()
        elif choice == "27":
            calc_capm()
        elif choice == "28":
            calc_beta()
        elif choice == "29":
            cli_sharpe_ratio()
        elif choice == "30":
            cli_efficient_frontier()
        elif choice == "31":
            cli_parametric_var()
        elif choice == "32":
            cli_conditional_var()
        elif choice == "33":
            cli_scale_var()
        elif choice == "34":
            cli_moving_average()
        elif choice == "35":
            cli_exponential_moving_average()
        elif choice == "36":
            cli_ar_model()
        elif choice == "37":
            cli_arima_model()
        elif choice == "38":
            ui()
        elif choice == "39":
            run_kalman_filter()
        elif choice == "40":
            run_random_forest()
        elif choice == "41":
            run_neural_network()
        elif choice == "42":
            run_garch_model()
        elif choice == "43":
            run_least_squares()
        elif choice == "44":
            run_r_squared()
        elif choice == "45":
            run_polynomial_regression()
        elif choice == "46":
            run_time_value_money()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
