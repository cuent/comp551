import numpy as np


class LinearClassifier():
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        raise Exception('I cannot train...')

    def pred(self, X):
        X = self.preprocess_input(X)
        return X.dot(self.w)

    def preprocess_input(self, X):
        X = X.reshape(X.shape[0], -1)
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((X, bias))
        return X


class LinearRegressionMSE(LinearClassifier):

    def fit(self, X, y):
        X = self.preprocess_input(X)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)[:, np.newaxis])
        return self


class LinearRegressionGD(LinearClassifier):

    def fit(self, X, y, lr=10e-6, tol=10e-9):
        X = self.preprocess_input(X)
        # init value of weights
        w_ = np.random.rand(X.shape[1], 1)
        self.w = np.ones((X.shape[1], 1))
        error = 1

        while (error > tol):
            dw = 2 * (X.T.dot(X).dot(w_) - X.T.dot(y[:, np.newaxis]))
            w_ -= lr * dw

            error = np.max(np.abs(self.w - w_))
            self.w = w_.copy()
        return self
