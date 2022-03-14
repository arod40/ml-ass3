from math import inf

import matplotlib.pyplot as plt
import numpy as np


def gradient_ascent(target, d, bounds, max_steps=10000, lr=0.1):
    lower, upper = bounds

    best_f = inf
    best_w = None
    f_prev = inf

    eps = 1e-8

    w = np.zeros((d, 1))

    it = 0
    while it < max_steps:
        it += 1

        f, gradient = target(w)
        w += lr * gradient

        if not (w >= lower).all() and (w <= upper).all():
            continue

        if abs(f_prev - f) < eps:
            break
        else:
            f_prev = f

        if f < best_f:
            best_w = w
            best_f = f

    return best_w, best_f, it, lr


def pocket_algorithm(X, y, max_iter=-1, w_init=None):
    def perceptron(x, w):
        val = x @ w
        ones = np.ones(val.shape)
        return (val >= 0) * ones - (val < 0) * ones

    w = w_init if w_init is not None else np.zeros((X.shape[1], 1))
    least_inc = inf
    best_w = None
    it = 0
    while max_iter == -1 or it < max_iter:
        it += 1
        inc_idxs = (perceptron(X, w) != y).squeeze()
        num_inc = inc_idxs.sum()

        if num_inc < least_inc:
            least_inc = num_inc
            best_w = w
        if num_inc == 0:
            break

        incorrect_x = X[inc_idxs][:1, :].transpose()
        incorrect_y = y[inc_idxs][0].squeeze()
        w += incorrect_y * incorrect_x
    return best_w, (perceptron(X, w) != y).sum()


def linear_regression(X, y):
    X_t = X.transpose()
    return np.linalg.inv(X_t @ X) @ X_t @ y


def logistic_regression(X, y, max_iter, lr):
    sigmoid = np.vectorize(lambda x: 1 / (1 + np.e ** (-x)))
    X_t = X.transpose()

    def ce(w):
        N = X.shape[0]
        f = np.log(sigmoid(X @ w)).sum() / N
        gradient = X_t @ (y * sigmoid(-y * (X @ w)))

        return f, gradient

    w, _, _, _, = gradient_ascent(
        ce, X.shape[1], (-inf, inf), max_steps=max_iter, lr=lr
    )
    return w


def plot_line(w, x1, x2, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    plt.plot([x1, x2], [y1, y2], color=color, label=label)


def get_data():
    c = 3 * np.random.randn(2, 2)

    a = np.array([1, 2, 3])
    a.squeeze

    y = np.concatenate([-1 * np.ones((50, 1)), np.ones((50, 1))])
    X = np.random.randn(100, 2) + c[((y.squeeze() + 3) / 2 - 1).astype("int64")]
    X = np.concatenate([np.ones((100, 1)), X], axis=1)

    return X, y


def add_outliers(y, ratio=0.1):
    y_outliers = np.copy(y)
    neg_y = (y == -1).squeeze()
    neg_y_total = neg_y.sum()
    outliers_idx = np.random.choice(
        range(neg_y_total), int(neg_y_total * ratio), replace=False
    )
    switch = np.ones((neg_y_total, 1)) * -1
    switch[outliers_idx] = 1
    y_outliers[neg_y] = switch

    return y_outliers


if __name__ == "__main__":
    X, y = get_data()
    y_ = y.squeeze()

    plt.xlim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)

    plt.scatter(X[y_ == 1, 1], X[y_ == 1, 2], marker="o", color="blue")
    plt.scatter(X[y_ == -1, 1], X[y_ == -1, 2], marker="x", color="red")

    w_lg = linear_regression(X, y)
    w_p, inc1 = pocket_algorithm(X, y, max_iter=10000)
    w_p2, inc2 = pocket_algorithm(X, y, max_iter=10000, w_init=w_lg)
    w_log = logistic_regression(X, y, max_iter=10000, lr=0.1)

    plot_line(
        list(w_lg), X[:, 1].min(), X[:, 1].max(), "blue", label="linear regression"
    )
    plot_line(list(w_p), X[:, 1].min(), X[:, 1].max(), "red", label="pocket algorithm")
    # plot_line(
    #     list(w_p2),
    #     X[:, 1].min(),
    #     X[:, 1].max(),
    #     "green",
    #     label="pocket with lin regression init",
    # )
    plot_line(
        list(w_log), X[:, 1].min(), X[:, 1].max(), "black", label="logistic regression"
    )
    plt.legend()
    plt.show()

    plt.xlim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)

    # y_outliers = add_outliers(y)
    # y_ = y_outliers.squeeze()
    # plt.scatter(X[y_ == 1, 1], X[y_ == 1, 2], marker="o", color="blue")
    # plt.scatter(X[y_ == -1, 1], X[y_ == -1, 2], marker="x", color="red")

    # y = y_outliers
    # w_lg = linear_regression(X, y)
    # w_p, inc1 = pocket_algorithm(X, y, max_iter=10000)
    # w_p2, inc2 = pocket_algorithm(X, y, max_iter=10000, w_init=w_lg)

    # plot_line(list(w_lg), X[:, 1].min(), X[:, 1].max(), "blue")
    # plot_line(list(w_p), X[:, 1].min(), X[:, 1].max(), "red")
    # plot_line(list(w_p2), X[:, 1].min(), X[:, 1].max(), "green")
    # plt.show()

