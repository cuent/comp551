from utils import *
from linear import *
import matplotlib.pyplot as plt


def more_points(n):
    f = lambda x: 2 * x + 3
    x, y = generate_1d_data(f, n, 30)

    # Execute LR
    lr = LinearRegressionMSE()
    lr.fit(x, y)
    y_pred = lr.pred(x)

    fig, ax = plot_data(x, y, y_pred)
    # ax.plot([np.min(x), np.max(x)], [f1(np.min(x)), f1(np.max(x))], 'r-', lw=1, label="Prediction")
    plt.legend(loc='best')
    fig.savefig('result/more_points.png')


def course_example():
    x = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])
    y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

    # Execute LR
    lr = LinearRegressionMSE()
    lr.fit(x, y)
    y_pred = lr.pred(x)

    # Linear equation
    m, b = lr.w[0, 0], lr.w[1, 0]
    f1 = lambda x: m * x + b

    # Plot results
    f, ax = plot_data(x, y, y_pred)
    ax.plot([np.min(x), np.max(x)], [f1(np.min(x)), f1(np.max(x))], 'r-', lw=1, label="Prediction")
    plt.legend(loc='best')
    f.savefig('result/course_example_mse.png')


def lr_gradient_descent():
    x = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])
    y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

    # Execute LR
    lr = LinearRegressionGD()
    lr.fit(x, y)
    y_pred = lr.pred(x)

    # Linear equation
    m, b = lr.w[0, 0], lr.w[1, 0]
    f1 = lambda x: m * x + b

    # Plot results
    f, ax = plot_data(x, y, y_pred)
    ax.plot([np.min(x), np.max(x)], [f1(np.min(x)), f1(np.max(x))], 'r-', lw=1, label="Prediction")
    plt.legend(loc='best')
    f.savefig('result/course_example_gd.png')


def features_linear_dependent():
    f1 = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])
    X = np.stack((f1, f1 + 2)).T
    y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

    lr = LinearRegressionMSE()
    try:
        lr.fit(X, y)
    except:
        print("Singular Matrix")


def polynomial_fit():
    # data
    x = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])

    X = np.stack((x ** 2, x)).T
    y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

    # model
    lr = LinearRegressionMSE()
    lr.fit(X, y)
    w = lr.w.flatten()

    # plotting
    f = lambda x: w[2] * x ** 2 + w[1] * x + w[0]

    fig, ax = plot_data(x, y, f(x))
    x.sort()
    ax.plot(x, f(x), 'k-', label="model")

    plt.legend()
    fig.savefig('result/course_example_polynimial.png')

if __name__ == "__main__":
    course_example()
    lr_gradient_descent()
    more_points(500)
    features_linear_dependent()
    polynomial_fit()
