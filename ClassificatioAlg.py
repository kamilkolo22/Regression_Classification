import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
from heapq import nsmallest
from time import time
from scipy.spatial.distance import mahalanobis


class KNeighbours:
    """K nearest neighbours algorithm with k param fit"""

    def __init__(self, k_neighbors=0, k_min=1, k_max=5, n_splits=5,
                 shuffle=True, metric='euler'):

        if metric != 'euler' and metric != 'mahalanobis':
            raise ValueError(f'Wrong metric param value: {metric}')

        self.k_neighbors = k_neighbors
        self.n_splits = n_splits
        self.k_max = k_max
        self.k_min = k_min
        self.shuffle = shuffle

        self.x_train = None
        self.y_train = None
        self.size = None
        self.dim = None
        self.k = None

        self.x_test = None
        self.size_test = None
        self.y_pred = None

        self.inv_corr_matrix = None
        self.metric = metric

    def fit(self, x_train, y_train, fast_fit=False):
        """Fit model by checking different k param values"""
        if self.metric == 'mahalanobis':
            self.inv_corr_matrix = np.linalg.inv(pd.DataFrame(x_train).corr())

        # check k values if it is set to zero
        if self.k_neighbors == 0:
            # Split data for cross validation
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
            k_array = []

            for k in range(self.k_min, self.k_max + 1):
                time_start = time()
                self.k = k
                scores = []

                for train_index, test_index in kf.split(x_train):
                    x_train_cv = x_train[train_index]
                    x_test_cv = x_train[test_index]
                    y_train_cv = y_train[train_index]
                    y_test_cv = y_train[test_index]

                    # fit model to cross validation set
                    if fast_fit:
                        kNN = KNeighborsClassifier(n_neighbors=self.k)
                        kNN.fit(x_train_cv, y_train_cv)
                        pr = kNN.predict(x_test_cv)
                    else:
                        pr = self.predict(x_test_cv, x_train_cv,
                                          y_train_cv)
                    ac_s = accuracy_score(y_test_cv, pr)
                    scores.append(ac_s)

                # Write lowest score from cross validation
                scores_min = min(scores)
                k_array.append(scores)
                print(
                    f'k: {k}, time: {time() - time_start}, score: {scores_min}')

            # Select k for which score value was highest
            self.k = max(list(enumerate(k_array)), key=lambda x: x[1])[0] + 1
            self.x_train = x_train
            self.y_train = y_train
        else:
            self.k = self.k_neighbors
            self.x_train = x_train
            self.y_train = y_train

    def predict(self, x_test, x_train=None, y_train=None):
        """Predict values using KNN algorithm and mahalanobis metric"""
        if x_train is not None:
            self.x_train = x_train
            self.y_train = y_train
            self.size = len(self.x_train)
            self.dim = len(self.x_train[0])

        self.x_test = x_test
        self.size_test = len(self.x_test)
        self.y_pred = []

        for v in self.x_test:
            distances = []
            for a, b in zip(self.x_train, self.y_train):
                # Chose metric
                if self.metric == 'euler':
                    dist = np.linalg.norm(a - v)
                else:
                    dist = mahalanobis(a, v, self.inv_corr_matrix)

                distances.append((dist, b))
            # Find k smallest distances
            smallest_distances = nsmallest(self.k, distances)

            # find mode for set of classes
            close = mode([x[1] for x in smallest_distances])
            self.y_pred.append(close)

        return self.y_pred
