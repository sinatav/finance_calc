import numpy as np
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
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
