import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    plt.plot(y_test[:50].values, label="True")
    plt.plot(predictions[:50], label="Predicted")
    plt.legend()
    plt.title("Random Forest Predictions")
    plt.show()