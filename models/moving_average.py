def moving_average(prices: list[float], window: int) -> list[float]:
    """Calculate simple moving average."""
    return [sum(prices[i:i+window])/window for i in range(len(prices)-window+1)]


def exponential_moving_average(prices: list[float], alpha: float) -> list[float]:
    """Calculate exponential moving average."""
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return ema
