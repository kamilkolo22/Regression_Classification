import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ReadData import *
from ClassificatioAlg import KNeighbours
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


def run_classification_alg():
    df = read_classification_data()
    df = df[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
             'int_memory', 'n_cores', 'ram', 'touch_screen', 'wifi', 'price_range']]
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # X_scaled = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        test_size=0.2)
    knn = KNeighbours(k_min=5, k_max=30)
    knn.fit(X_train, y_train, fast_fit=True)
    pred = knn.predict(X_test)
    print(f'Otrzymaliśmy accuracy: {accuracy_score(y_test, pred)} przy k '
          f'najbliższych sąsiadów równym: {knn.k}')
    print(f'Confusion matrix: \n{pd.DataFrame(confusion_matrix(y_test, pred))}')


def run_regression():
    df = read_regression_data()
    # scaler = StandardScaler()
    # scaler.fit(df)
    # df_scaled = pd.DataFrame(scaler.transform(df))
    # df_scaled.columns = df.columns
    X_train, X_test, y_train, y_test = \
        train_test_split(df[['Var_av', 'Var_LT', 'Var_mass']], df[['Target']],
                         test_size=0.3, random_state=101)
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test).flatten()
    MAE = mean_absolute_error(y_test, test_predictions)
    MSE = mean_squared_error(y_test, test_predictions)
    RMSE = np.sqrt(MSE)
    print(f'MAE: {MAE}\nMSE: {MSE}\nRMSE: {RMSE}\n'
          f'Target mean: {df["Target"].mean()}\n'
          f'Target Std dev: {df["Target"].std()}')
    y_test = np.array(y_test).flatten()
    residuals = y_test - test_predictions
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    sns.scatterplot(x=y_test, y=test_predictions)
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False,
             scaley=False)
    plt.show()
    sns.kdeplot(residuals)
    plt.show()


if __name__ == '__main__':
    run_classification_alg()

    # run_regression()
