import numpy as np
from sys import platform as sys_pf

if sys_pf == 'darwin':
    # solve crashing issue https://github.com/MTG/sms-tools/issues/36
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class LinearClassifier():
    """ Base class for linear classifiers. """

    def __init__(self, lr=None):
        self.w = None
        self.lr = lr

    def fit(self, X, y):
        """Builds a model given the input data.
        :param X: input data. Values must be numeric.
        :param y: target variable.
        :return:
        """
        raise Exception('I cannot train...')

    def pred(self, X):
        """Predicts values for the input data given.
        :param X: input data.
        """
        X = self.preprocess_input(X)
        return X.dot(self.w)

    def preprocess_input(self, X, y=None):
        X = X.reshape(X.shape[0], -1)
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((X, bias))
        if y is None:
            return X
        y = y if len(y.shape) > 1 else y[:, np.newaxis]
        return X, y


class LinearRegressionMSE(LinearClassifier):
    """Linear regression using closed-form solution."""

    def fit(self, X, y):
        X, y = self.preprocess_input(X, y)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return self


class LinearRegressionGD(LinearClassifier):
    """Linear regression using gradient descent."""

    def __init__(self, lr):
        super().__init__(lr)
        self.step = 0
        self.errors = []
        self.error = 1
        # TODO: save the rate of change of the lr

    def fit(self, X, y, tol=1e-5, init_const=False, max_iter=1000, verbose=False):
        # Preprocess inut data (i.e., add bias)
        X, y = self.preprocess_input(X, y)
        # Init value of weights
        self.w = np.random.rand(X.shape[1], 1)
        if init_const:
            self.w = np.zeros((X.shape[1], 1))

        n = X.shape[0]
        # keep this variables constant, so we don't have to compute this in very iteration of GD
        const1 = X.T.dot(y)
        const2 = X.T.dot(X)

        while self.error > tol and self.step < max_iter:
            w_old = self.w.copy()

            dw = 2 * (const2.dot(self.w) - const1)
            self.w -= self.lr.compute(dw / n)  # scale gradient by n to avoid the gradient vanishing

            self.error = np.linalg.norm(np.abs(w_old - self.w), 2)
            # some variables for plotting and logs
            self.errors.append(self.error)
            self.step += 1

            if self.step % 100 == 0 and verbose:
                print('\r > Iteration {}: error {}'.format(str(self.step), str(self.error)), end=".")
                print()
        return self

    def plot_error(self):
        it = list(range(len(self.errors)))
        plt.plot(it, self.errors)
        plt.title("Error vs Iteration")
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

    def __str__(self):
        return "{},error={},step={}".format(self.lr, self.error, self.step)
