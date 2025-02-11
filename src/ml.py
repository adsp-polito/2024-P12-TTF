from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_lr(X_train: np.ndarray, y_train: np.ndarray, X_val, y_val, path: str):
    import joblib
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2])) # reshape the windows as rows of features
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    joblib.dump(regressor, path)

def train_rf(X_train: np.ndarray, y_train: np.ndarray, X_val, y_val, path: str):
    import joblib
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2])) # reshape the windows as rows of features
    
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    joblib.dump(regressor, path)

def predict(X_test: np.ndarray, path):
    import joblib
    import os
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2])) # reshape the windows as rows of features

    if not os.path.isfile(path):
        raise ValueError(f"The regressor at path {path} does not exist.")

    regressor: LinearRegression | RandomForestRegressor = joblib.load(path)
    y_pred = regressor.predict(X_test)
    return y_pred