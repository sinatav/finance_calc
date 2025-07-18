def scale_var(var_1day: float, days: int) -> float:
    """Scale 1-day VaR to multiple days using square root of time."""
    return var_1day * np.sqrt(days)
