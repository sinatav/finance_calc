import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def polynomial_fit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)

    plt.scatter(x, y, label='Data')
    plt.plot(x, y_pred, 'r', label=f'Poly deg {degree}')
    plt.title("Polynomial Regression")
    plt.legend()
    plt.show()
    return coeffs
