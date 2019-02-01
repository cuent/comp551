import numpy as np


class LinearClassifier():
    """ Base class for linear classifiers """

    def __init__(self, lr=None):
        self.w = None
        self.lr = lr

    def fit(self, X, y):
        """
        Builds a model given the input data.
        :param X:
        :param y:
        :return:
        """
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
    def __init__(self, lr):
        super().__init__(lr)
        self.step = 0

    def fit(self, X, y, tol=10e-9):
        X = self.preprocess_input(X)
        # init value of weights
        w_ = np.random.rand(X.shape[1], 1)
        self.w = np.ones((X.shape[1], 1))
        error = 1

        while (error > tol):
            dw = 2 * (X.T.dot(X).dot(w_) - X.T.dot(y[:, np.newaxis]))
            w_ -= self.lr.compute(dw)

            error = np.linalg.norm(np.abs(self.w - w_), 2)
            self.w = w_.copy()
            self.step += 1
        print(self.step)
        return self


class LearningRate():
    def __init__(self, lr=10e-6):
        self.lr = lr

    def compute(self, dw):
        return self.lr * dw


class Decay(LearningRate):
    def __init__(self, lr=10e-3, b=10e-3):
        super().__init__(lr)
        self.b = b
        self.decay_step = 1

    def compute(self, dw):
        decay = self.lr / (1 + self.b * self.decay_step)
        self.decay_step += 1
        return decay * dw


class Momentum(LearningRate):
    def __init__(self, lr=10e-3, b=0.9):
        super().__init__(lr)
        self.b = b
        self.momentum = 0

    def compute(self, dw):
        self.momentum = (self.momentum + dw) / 2
        grad = self.b * self.momentum + (1 - self.b) * dw
        return self.lr * grad


def tokenize(str):
    '''Tokenize text: Lower case and divide into words.'''
    return str.lower().split(" ")


def build_dictionary(train_data, dict_size=160):
    '''Build dictionary: Find words and count frequency.'''
    # Create an output dictionary to store the word frequencies
    dic = {}
    out = []

    # Loop over each sample of the training set
    for sample in train_data:
        # Split the sample text into lowercase words
        tokens = tokenize(sample['text'])
        # Accumulate words frequencies into the dictionary
        for token in tokens:
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1

    # Select the most frequent words only
    i = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        out.append(w)
        if i == dict_size:
            break

    # Return list of most frequent words
    return out


def str2vec(str, dic):
    vec = [0] * len(dic)
    words = tokenize(str)
    for word in words:

