import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ReadData import read_regression_data, read_classification_data
from ClassificatioAlg import KNeighbours
from RegressionAlg import fit_regression, split_data


def run_classification_alg():
    df = read_classification_data()
    df = df[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
             'int_memory', 'n_cores', 'ram', 'touch_screen', 'wifi', 'price_range']]
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # scaler = StandardScaler()
    # scaler.fit(X)
    # X_scaled = scaler.transform(X)
    # X_scaled = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = KNeighbours(
        # k_neighbors=25,
        metric='mahalanobis',
        k_min=5,
        k_max=30
    )
    knn.fit(X_train, y_train, fast_fit=False)
    pred = knn.predict(X_test)

    print(f'Otrzymaliśmy accuracy: {accuracy_score(y_test, pred)} przy k '
          f'najbliższych sąsiadów równym: {knn.k}')
    print(f'Confusion matrix: \n{pd.DataFrame(confusion_matrix(y_test, pred))}')


def run_regression():
    df = read_regression_data()

    # split data
    x_train, x_test, x_validate, y_train, y_test, y_validate = \
        split_data(df[['Var_av', 'Var_LT', 'Var_mass']], df[['Target']])

    results = fit_regression(x_train, y_train, x_test, y_test)

    print(f'Results for different models: \n{results}')
    kernel = input('Chose kernel\n')
    epsilon = float(input('Chose epsilon\n'))

    model = make_pipeline(StandardScaler(), SVR(kernel=kernel, epsilon=epsilon))
    model.fit(x_train, y_train)
    pred = model.predict(x_validate).flatten()

    MAE = mean_absolute_error(y_validate, pred)
    MSE = mean_squared_error(y_validate, pred)
    R2 = r2_score(y_validate, pred)

    print(f'MAE: {MAE}\nMSE: {MSE}\nR2: {R2}\n'
          f'Target mean: {df["Target"].mean()}\n'
          f'Target Std dev: {df["Target"].std()}')

    y_validate = np.array(y_validate).flatten()
    residuals = y_validate - pred

    sns.scatterplot(x=y_validate, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    sns.scatterplot(x=y_validate, y=pred)
    x_points = y_points = plt.xlim()
    plt.plot(x_points, y_points, linestyle='--', color='k', lw=3, scalex=False,
             scaley=False)
    plt.show()

    sns.kdeplot(residuals)
    plt.show()


if __name__ == '__main__':
    # run_classification_alg()
    run_regression()
