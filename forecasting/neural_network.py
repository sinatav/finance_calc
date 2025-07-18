import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_neural_network(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Neural Network RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    plt.plot(y_test[:50].values, label="True")
    plt.plot(predictions[:50], label="Predicted")
    plt.legend()
    plt.title("Neural Network Predictions")
    plt.show()
