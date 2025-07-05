import math


class ZeroCouponBond:
    def __init__(self, face_value: float, price: float, time: float):
        self.F = face_value
        self.P = price
        self.t = time

    def ytm(self):
        # YTM = (F/P)^(1/t) - 1
        return (self.F / self.P) ** (1 / self.t) - 1


class CouponBearingBond:
    def __init__(self, face_value: float, coupon_rate: float, price: float, time: float, freq: int):
        self.F = face_value
        self.c = coupon_rate
        self.P = price
        self.t = time
        self.freq = freq  # number of coupon payments per year

    def ytm_simplified(self):
        # simplified YTM = (Coupon + (F - P) / t) / ((F + P)/2)
        C = self.F * self.c / self.freq
        return (C + (self.F - self.P) / self.t) / ((self.F + self.P) / 2)

    def ytm_continuous(self, tolerance=1e-6, max_iter=100):
        """
        Calculates YTM assuming continuous compounding using Newton-Raphson.
        Formula:
        P = sum_i (C * e^(-y * (t_i - t))) + F * e^(-y * (T - t))
        We assume current time t = 0 for simplicity
        """

        n = int(self.freq * self.t)
        C = self.F * self.c / self.freq
        times = [(i + 1) / self.freq for i in range(n)]  # e.g., [0.5, 1.0, ..., T]

        def f(y):
            total = sum(C * math.exp(-y * t_i) for t_i in times)
            total += self.F * math.exp(-y * self.t)
            return total - self.P

        def df(y):
            total = sum(-t_i * C * math.exp(-y * t_i) for t_i in times)
            total += -self.t * self.F * math.exp(-y * self.t)
            return total

        # Initial guess
        y = self.ytm_simplified()

        for _ in range(max_iter):
            f_val = f(y)
            df_val = df(y)
            if abs(df_val) < 1e-10:
                break
            y_new = y - f_val / df_val
            if abs(y_new - y) < tolerance:
                return y_new
            y = y_new

        return y

    def ytm_original(self, tolerance=1e-6, max_iter=100):
        """
        Uses Newton-Raphson method to solve the YTM from the bond price formula:
        P = sum_{i=1}^{n} C/(1+y/f)^{i} + F/(1+y/f)^n
        where n = freq * t, C = coupon payment, y = YTM
        """

        n = int(self.freq * self.t)
        C = self.F * self.c / self.freq

        # Initial guess
        y = self.ytm_simplified()

        def f(ytm):
            total = 0
            for i in range(1, n + 1):
                total += C / ((1 + ytm / self.freq) ** i)
            total += self.F / ((1 + ytm / self.freq) ** n)
            return total - self.P

        def df(ytm):
            total = 0
            for i in range(1, n + 1):
                total -= (i * C) / (self.freq * (1 + ytm / self.freq) ** (i + 1))
            total -= (n * self.F) / (self.freq * (1 + ytm / self.freq) ** (n + 1))
            return total

        for _ in range(max_iter):
            f_val = f(y)
            df_val = df(y)
            if df_val == 0:
                break
            y_new = y - f_val / df_val
            if abs(y_new - y) < tolerance:
                return y_new
            y = y_new

        return y  # return last value if no convergence
