import sys

# Risk and Forecasting Models
from models.scaling_var import scale_var
from models.moving_average import calculate_sma, calculate_ema
from models.ar_model import fit_ar_model
from models.ma_model import fit_ma_model
from models.arima_model import fit_arima_model

def run_scale_var():
    try:
        var_1_day = float(input("Enter 1-day VaR: "))
        days = int(input("Enter number of days to scale to: "))
        scaled = scale_var(var_1_day, days)
        print(f"Scaled {days}-day VaR: {scaled:.4f}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")

def run_sma():
    try:
        prices = list(map(float, input("Enter price data (comma-separated): ").split(",")))
        window = int(input("Enter SMA window size: "))
        result = calculate_sma(prices, window)
        print(f"Simple Moving Average: {result}")
    except ValueError:
        print("Invalid input.")

def run_ema():
    try:
        prices = list(map(float, input("Enter price data (comma-separated): ").split(",")))
        span = int(input("Enter EMA span: "))
        result = calculate_ema(prices, span)
        print(f"Exponential Moving Average: {result}")
    except ValueError:
        print("Invalid input.")

def run_ar_model():
    try:
        series = list(map(float, input("Enter time series data (comma-separated): ").split(",")))
        lags = int(input("Enter number of lags (p): "))
        fit_ar_model(series, lags)
    except ValueError:
        print("Invalid input.")

def run_ma_model():
    try:
        series = list(map(float, input("Enter time series data (comma-separated): ").split(",")))
        q = int(input("Enter MA order (q): "))
        fit_ma_model(series, q)
    except ValueError:
        print("Invalid input.")

def run_arima_model():
    try:
        series = list(map(float, input("Enter time series data (comma-separated): ").split(",")))
        p = int(input("Enter AR order (p): "))
        d = int(input("Enter differencing order (d): "))
        q = int(input("Enter MA order (q): "))
        fit_arima_model(series, p, d, q)
    except ValueError:
        print("Invalid input.")

def ui():
    while True:
        print("\n--- Quant Finance CLI ---")
        print("1. Scale VaR")
        print("2. Simple Moving Average (SMA)")
        print("3. Exponential Moving Average (EMA)")
        print("4. Fit Autoregressive (AR) Model")
        print("5. Fit Moving Average (MA) Model")
        print("6. Fit ARIMA Model")
        print("0. Exit")

        try:
            choice = int(input("Select an option: "))
        except ValueError:
            print("Invalid input. Try again.")
            continue

        if choice == 0:
            print("Exiting...")
            sys.exit()

        elif choice == 1:
            run_scale_var()
        elif choice == 2:
            run_sma()
        elif choice == 3:
            run_ema()
        elif choice == 4:
            run_ar_model()
        elif choice == 5:
            run_ma_model()
        elif choice == 6:
            run_arima_model()
        else:
            print("Invalid choice. Please select a valid option.")
