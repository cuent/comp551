from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.sparse import issparse


class BernoulliNaiveBayes(BaseEstimator, ClassifierMixin):
    '''
    Naive Bayes classifier for binary data. Uses bernoulli distribution for features and target must be binary as well.
    '''

    def __init__(self, smooth=1):
        '''
        :param smooth: smoothing parameter by default uses laplace smoothing (smooth=1)
        '''
        self.smooth = smooth

    def fit(self, X, y):
        '''
        Trains classifier according with training data and labels.
        :param X: training data [n:instances, m:features]
        :param y: target variable [n:instances, 1]
        :return: self
        '''
        # TODO: use sparse matrix to improve calculations.
        if issparse(X):
            raise TypeError('BernoulliNaiveBayes does not support sparse input.')

        n, m = X.shape
        y = y[:, np.newaxis] if len(y.shape) == 1 else y
        assert y.shape == (n, 1), "Incorrect target dimension"
        assert (np.unique(X) == [0, 1]).all(), "X should be binary"
        assert (np.unique(y) == [0, 1]).all(), "y should be binary"

        clazz = np.array([0, 1])

        self.conditional = np.zeros((len(clazz), m))
        self.prior = np.zeros(len(clazz))

        for i, c in enumerate(clazz):
            nc = len(y[y == c])
            self.prior[i] = nc / n
            for j in range(m):
                class_only = X[np.argwhere(y.reshape(n) == c), j]  # y = c
                self.conditional[i, j] = (len(class_only[class_only == 1]) + self.smooth) / (
                        nc + 2 * self.smooth)  # y=c & x_j=1

        return self

    def predict(self, X):
        '''
        Make predictions with given data.
        :param X: input data [n:instances, m:features]
        :return: predictions [n:instances, 1]
        '''
        wj1 = np.log(self.conditional[1]) - np.log(self.conditional[0])
        wj0 = np.log(1 - self.conditional[1]) - np.log(1 - self.conditional[0])
        boundary = (wj0 * (1 - X) + wj1 * X).sum(axis=1) + np.log(self.prior[1]) - np.log(self.prior[0])
        return np.where(boundary > 0, 1, 0)[:, np.newaxis]
