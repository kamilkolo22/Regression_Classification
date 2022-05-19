import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from statistics import mode
from collections import Counter


class KNeighbours:

    def __init__(self, k_neighbors=0, k_max=5, n_splits=5, shuffle=True):

        self.k_neighbors = k_neighbors
        self.n_splits = n_splits
        self.k_max = k_max
        self.shuffle = shuffle

        self.X_train = None
        self.y_train = None
        self.size = None
        self.dim = None
        self.k = None

        self.X_test = None
        self.size_test = None
        self.y_test = None

    # def __prefit(self, X_train, y_train):
    #     self.X_train = X_train
    #     self.y_train = y_train
    #     self.size = len(self.X_train)
    #     self.dim = len(self.X_train[0])

    def fit(self, x_train, y_train):

        # self.__prefit(X_train, y_train)

        # if self.k_max == 0:
        #     self.k_max = self.size

        if self.k_neighbors == 0:
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
            k_array = np.full(self.k_max, 0.0)
            index_k = 0

            for k in range(1, self.k_max + 1):
                self.k = k
                scores = np.full(self.n_splits, 0.0)
                indexCV = 0
                # kf.get_n_splits(self.X_train)

                for train_index, test_index in kf.split(x_train):
                    X_trainCV, X_testCV = x_train[train_index], \
                                          x_train[test_index]
                    y_trainCV, y_testCV = y_train[train_index], \
                                          y_train[test_index]

                    # self.__prefit(X_trainCV, y_trainCV)

                    pr = self.predict(X_testCV, X_trainCV, y_trainCV)
                    print(y_testCV)
                    print(pr)
                    acS = accuracy_score(y_testCV, pr)
                    scores[indexCV] = acS
                    indexCV += 1
                    # Powrót do wejściowego zbioru treningowego:
                    # self.__prefit(X_train, y_train)

                scores = np.sort(scores)
                print(f'scores: {scores}')
                k_array[index_k] = scores[0]
                index_k += 1

            self.k = np.argmax(k_array) + 1
            self.X_train = x_train
            self.y_train = y_train
        else:
            self.k = self.k_neighbors
            self.X_train = x_train
            self.y_train = y_train

    def predict(self, x_test, x_train=None, y_train=None):

        if x_train is not None:
            self.X_train = x_train
            self.y_train = y_train
            self.size = len(self.X_train)
            self.dim = len(self.X_train[0])

        self.X_test = x_test
        self.size_test = len(self.X_test)
        self.y_test = np.full(self.size_test, 0.0)

        # distances = np.full((self.size, 2), 0.0)
        distances = []
        index_q = 0

        for q in self.X_test:
            index = 0
            for a in self.X_train:
                ## TODO add mahalonobis metric
                dist = np.linalg.norm(a - q)
                distances.append((dist, self.y_train[index]))
                # distances[index, 0] = dist
                # distances[index, 1] = self.y_train[index]
                index += 1

            # distances = distances[distances[:, 0].argsort()]
            distances.sort(key=lambda x: x[0])
            SmallestDistances = distances[0:self.k]

            # cl = mode(SmallestDistances[:,1])- ta funkcja niestety nie radzi sobie z więcej niż jedną modą
            # cl = Counter(SmallestDistances[:, 1]).most_common(1)[0][0]
            cl = Counter([x[1] for x in SmallestDistances]).most_common(1)[0][0]
            self.y_test[index_q] = cl
            index_q += 1
        return self.y_test

# from ReadData import *
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# df = read_classification_data()
#
#
# X = df.to_numpy()[:, :-1]
# y = df.to_numpy()[:, -1]
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# knn = KNeighbours()
# knn.fit(X_train, y_train)
#
# pred = knn.predict(X_test)
#
# print(f'Otrzymaliśmy accuracy: {accuracy_score(y_test, pred)} przy k '
#         f'najbliższych sąsiadów równym: {knn.k}')