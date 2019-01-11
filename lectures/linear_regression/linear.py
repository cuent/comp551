import numpy as np


class LinearRegression():
    def __init__(self):
        a = 1

    def fit(self, X, y):
        X = self.__preprocess_input(X)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)[:, np.newaxis])
        return self

    def pred(self, X):
        X = self.__preprocess_input(X)
        return X.dot(self.w)

    def __preprocess_input(self, X):
        X = X.reshape(X.shape[0], -1)
        bias = np.ones((X.size, 1))
        X = np.hstack((X, bias))
        return X
