import numpy as np
from naive_bayes import BernoulliNaiveBayes


if __name__ == '__main__':
    np.random.seed(60)
    X = np.random.randint(2, size=(6, 15))
    y = np.random.randint(2, size=(6, 1))

    nv = BernoulliNaiveBayes()
    nv.fit(X, y)
    y_pred = np.ravel(nv.predict(X))


    print("original", np.ravel(y))
    print("predictions", y_pred)
