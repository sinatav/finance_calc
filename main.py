from interests import SimpleInterest, CompoundInterest, ContinuousCompounding
from bonds import ZeroCouponBond, CouponBearingBond
from options import Option, plot_greeks_vs_price
from pricing.monte_carlo import monte_carlo_option_price
from pricing.binomial_tree import binomial_tree_option_price
from pricing.pde_solver import crank_nicolson_option_price
from pricing.exotic_options import asian_option_price, digital_option_price


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
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
