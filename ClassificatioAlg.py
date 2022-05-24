import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from statistics import mode
from heapq import nlargest
from time import time
from scipy.spatial.distance import mahalanobis
from collections import Counter


class KNeighbours:
    def __init__(self, k_neighbors=0, k_min=1, k_max=5, n_splits=5, shuffle=True):

        self.k_neighbors = k_neighbors
        self.n_splits = n_splits
        self.k_max = k_max
        self.k_min = k_min
        self.shuffle = shuffle

        self.X_train = None
        self.y_train = None
        self.size = None
        self.dim = None
        self.k = None

        self.X_test = None
        self.size_test = None
        self.y_test = None

        self.inv_corr_matrix = None

    def fit(self, x_train, y_train, fast_fit=False):
        """Fit model by checking different k param values"""
        self.inv_corr_matrix = np.linalg.inv(pd.DataFrame(x_train).corr())

        if self.k_neighbors == 0:
            # Split data for cross validation
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
            k_array = []

            for k in range(self.k_min, self.k_max + 1):
                time_start = time()
                self.k = k
                scores = []

                for train_index, test_index in kf.split(x_train):
                    X_trainCV, X_testCV = x_train[train_index], \
                                          x_train[test_index]
                    y_trainCV, y_testCV = y_train[train_index], \
                                          y_train[test_index]

                    if fast_fit:
                        kNN = KNeighborsClassifier(n_neighbors=self.k)
                        kNN.fit(X_trainCV, y_trainCV)
                        pr = kNN.predict(X_testCV)
                    else:
                        pr = self.predict(X_testCV, X_trainCV, y_trainCV)
                    acS = accuracy_score(y_testCV, pr)
                    scores.append(acS)

                # Write lowest score from cross validation
                scores.sort()
                k_array.append(scores[0])
                print(f'k: {k}, time: {time() - time_start}, score: {scores[0]}')

            # Select k for which score value was highest
            self.k = max(list(enumerate(k_array)), key=lambda x: x[1])[0] + 1
            self.X_train = x_train
            self.y_train = y_train
        else:
            self.k = self.k_neighbors
            self.X_train = x_train
            self.y_train = y_train

    def predict(self, x_test, x_train=None, y_train=None):
        """Predict values using KNN algorithm and mahalonobis metric"""
        if x_train is not None:
            self.X_train = x_train
            self.y_train = y_train
            self.size = len(self.X_train)
            self.dim = len(self.X_train[0])

        self.X_test = x_test
        self.size_test = len(self.X_test)
        self.y_test = []
        distances = []

        for q in self.X_test:
            for a, b in zip(self.X_train, self.y_train):
                # dist = np.linalg.norm(a - q)
                dist = mahalanobis(a, q, self.inv_corr_matrix)
                distances.append((dist, b))

            SmallestDistances = nlargest(self.k, distances)

            cl = mode([x[1] for x in SmallestDistances])
            self.y_test.append(cl)
        return self.y_test

## This part is for debugging, can be deleted later
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
