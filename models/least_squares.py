import numpy as np
import matplotlib.pyplot as plt

def least_squares_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"Estimated line: y = {m:.4f}x + {c:.4f}")

    plt.scatter(x, y, label="Data")
    plt.plot(x, m*x + c, 'r', label="Fitted line")
    plt.legend()
    plt.title("Least Squares Regression")
    plt.show()
