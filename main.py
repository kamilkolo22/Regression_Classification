import pandas as pd
from ReadData import *
from ClassificatioAlg import KNeighbours
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # df = read_regression_data()

    df = read_classification_data()

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                        test_size=0.2)

    knn = KNeighbours(k_neighbors=5)
    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    print(f'Otrzymaliśmy accuracy: {accuracy_score(y_test, pred)} przy k '
          f'najbliższych sąsiadów równym: {knn.k}')
