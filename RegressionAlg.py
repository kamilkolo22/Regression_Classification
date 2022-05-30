import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_data(x, y, test_size=0.3, validate_size=0.1):
    """Split data to three sets, train, test and validation"""
    # Get test sets
    y = np.array(y).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size)

    # Run train_test_split again to get train and validate sets
    post_split_validate_size = validate_size / (1 - test_size)
    x_train, x_validate, y_train, y_validate = \
        train_test_split(x_train, y_train, test_size=post_split_validate_size)

    return x_train, x_test, x_validate, y_train, y_test, y_validate


def fit_regression(x_train, y_train, x_test, y_test):
    """Check different params in SVR model"""
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    epsilons = [0.05, 0.1, 0.25, 0.5]

    results = []
    for kernel in kernels:
        for epsilon in epsilons:
            repressor = make_pipeline(StandardScaler(),
                                      SVR(kernel=kernel, epsilon=epsilon))
            repressor.fit(x_train, y_train)
            pred = repressor.predict(x_test)

            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            results.append([kernel, epsilon, mae, mse, r2])

    results = pd.DataFrame(results, columns=['kernel', 'epsilon',
                                             'MAE', 'MSE', 'R2'])
    return results
