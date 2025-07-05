import math


class SimpleInterest:
    def __init__(self, principal: float, rate: float, time: float):
        self.P = principal
        self.r = rate
        self.t = time

    def calculate(self):
        return self.P * self.r * self.t


class CompoundInterest:
    def __init__(self, principal: float, rate: float, time: float, n: int):
        self.P = principal
        self.r = rate
        self.t = time
        self.n = n  # compounding frequency per year

    def calculate(self):
        return self.P * ((1 + self.r / self.n) ** (self.n * self.t) - 1)


class ContinuousCompounding:
    def __init__(self, principal: float, rate: float, time: float):
        self.P = principal
        self.r = rate
        self.t = time

    def calculate(self):
        return self.P * (math.exp(self.r * self.t) - 1)
