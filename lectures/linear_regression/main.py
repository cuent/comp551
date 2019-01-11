from utils import *
from linear import *
import matplotlib.pyplot as plt


def more_points(n):
    f = lambda x: 2 * x + 3
    x, y = generate_data(f, n)

    # Execute LR
    lr = LinearRegression()
    a = lr.fit(x, y)
    y_pred = lr.pred(x)

    fig, ax = plot_data(x, y, y_pred)
    # ax.plot([np.min(x), np.max(x)], [f1(np.min(x)), f1(np.max(x))], 'r-', lw=1, label="Prediction")
    plt.legend(loc='best')
    fig.savefig('result/more_points.png')


def course_example():
    x = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])
    y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

    # Execute LR
    lr = LinearRegression()
    a = lr.fit(x, y)
    y_pred = lr.pred(x)

    # Linear equation
    m, b = lr.w[0, 0], lr.w[1, 0]
    f1 = lambda x: m * x + b

    # Plot results
    f, ax = plot_data(x, y, y_pred)
    ax.plot([np.min(x), np.max(x)], [f1(np.min(x)), f1(np.max(x))], 'r-', lw=1, label="Prediction")
    plt.legend(loc='best')
    f.savefig('result/course_example.png')


if __name__ == "__main__":
    course_example()
    more_points(500)
