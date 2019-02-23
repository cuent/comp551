import numpy as np
import unittest
from src.naive_bayes import BernoulliNaiveBayes


class TestBNN(unittest.TestCase):

    def testBool(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y = np.array([[0], [0], [1], [1]])

        X_new = np.array([[1, 0], [1, 1]])
        y_exact = np.array([[0], [1]])

        nv = BernoulliNaiveBayes()
        nv.fit(X, y)
        y_pred = nv.predict(X_new)

        self.assertTrue((y_pred == y_exact).all())

    def testAgainstSklearn(self):
        X = np.random.randint(2, size=(6, 500))
        y = np.random.randint(2, size=(6, 1))

        from sklearn.naive_bayes import BernoulliNB

        skl_nb = BernoulliNB()
        skl_nb.fit(X, np.ravel(y))
        y_sklearn = skl_nb.predict(X)

        nv = BernoulliNaiveBayes()
        nv.fit(X, y)
        y_pred = np.ravel(nv.predict(X)).T

        self.assertTrue((y_sklearn == y_pred).all())
