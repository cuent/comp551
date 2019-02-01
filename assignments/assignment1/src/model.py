from data_process import load_data
import metric
import learning_rate as lr
import linear_regression as classifier
import time
import pickle
import numpy as np
import pandas as pd


def train_model(data_batch, init, learning_rates, momentum, decay):
    min_err = np.inf
    best = None

    # hyperparameters
    learning_rate_objects = []

    summary = pd.DataFrame(
        columns=['algorithm', 'mse train', 'mse val', 'init zero', 'iterations', 'time', 'lr method', 'lr', 'b',
                 'model'])

    # Load the preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_batch)

    # closed-form
    mse = classifier.LinearRegressionMSE()

    try:
        start = time.time()
        mse.fit(X_train, y_train)
        end = time.time()
    except Exception:
        print("Ignoring closed-form bc of singular matrix")
    else:
        y_pred = mse.pred(X_train)
        y_pred_val = mse.pred(X_val)
        print("Closed-form mse:{} - mse_val:{} (time={})".format(metric.mse(y_train, y_pred),
                                                                 metric.mse(y_val, y_pred_val), end - start))

        min_err = metric.mse(y_val, y_pred_val)
        best = mse

        summary = summary.append(
            {'algorithm': 'closed-form', 'mse train': metric.mse(y_train, y_pred), 'mse val': metric.mse(y_val, y_pred_val),
             'time': end - start, 'model': data_batch}, ignore_index=True)

    # learning rates
    for l in learning_rates:
        learning_rate_objects.append(lr.LearningRate(lr=l))
        for d in decay:
            learning_rate_objects.append(lr.Decay(lr=l, b=d))
        for m in momentum:
            learning_rate_objects.append(lr.Momentum(lr=l, b=m))

    # gradient descent
    for initialization in init:  # random vs zeros
        for lr_obj in learning_rate_objects:
            gd = classifier.LinearRegressionGD(lr_obj)

            start = time.time()
            gd.fit(X_train, y_train, init_const=initialization)
            end = time.time()

            y_pred = gd.pred(X_train)
            y_pred_val = gd.pred(X_val)
            print("Gradient Descent mse:{} - mse_val:{} (init={},iterations={},time={},{})".format(
                metric.mse(y_train, y_pred),
                metric.mse(y_val, y_pred_val),
                initialization, gd.step, end - start, lr_obj))

            if metric.mse(y_val, y_pred_val) < min_err:
                min_err = metric.mse(y_val, y_pred_val)
                best = gd

            summary = summary.append(
                {'algorithm': 'gradient', 'mse train': metric.mse(y_train, y_pred),
                 'mse val': metric.mse(y_val, y_pred_val),
                 'time': end - start, 'init zero': initialization, 'iterations': gd.step, 'model': data_batch,
                 'lr': lr_obj.lr, 'b': lr_obj.b if hasattr(lr_obj, 'b') else None,
                 'lr method': lr_obj.__class__.__name__},
                ignore_index=True)

    print("Best classifier achieved an error of", min_err)
    print(best)
    if isinstance(best, classifier.LinearRegressionGD):
        best.plot_error()

    return best, summary


def save_model(model):
    pickle.dump(model, open('model/model-{}'))
