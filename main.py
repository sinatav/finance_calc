from interests import SimpleInterest, CompoundInterest, ContinuousCompounding
from bonds import ZeroCouponBond, CouponBearingBond


def main_menu():
    print("\nFinance Calculator Menu:")
    print("1. Simple Interest")
    print("2. Compound Interest")
    print("3. Continuous Compounding")
    print("4. Zero-Coupon Bond YTM")
    print("5. Coupon-Bearing Bond YTM (Simplified)")
    print("6. Coupon-Bearing Bond YTM (Original)")
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
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
