import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

def run_kalman_filter_(observations):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(observations, n_iter=10)
    (filtered_state_means, _) = kf.filter(observations)

    plt.plot(observations, label="Observations")
    plt.plot(filtered_state_means, label="Kalman Filter")
    plt.legend()
    plt.title("Kalman Filter Result")
    plt.show()