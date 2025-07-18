def scale_var(var_1day: float, days: int) -> float:
    """
    Scale 1-day VaR to T-day VaR assuming independence and normality.
    VaR_T = VaR_1day * sqrt(T)
    """
    from math import sqrt
    return var_1day * sqrt(days)
