def future_value(pv, rate, periods):
    return pv * (1 + rate)**periods

def present_value(fv, rate, periods):
    return fv / (1 + rate)**periods

def annuity_value(payment, rate, periods):
    return payment * ((1 - (1 + rate)**-periods) / rate)

def perpetuity_value(payment, rate):
    return payment / rate
