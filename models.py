from math import inf

import numpy as np

from utils import timing

sigmoid = np.vectorize(
    lambda x: 1 / (1 + np.e ** (-x)) if x > 0 else 1 - 1 / (1 + np.e ** (x))
)


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

    return best_w


@timing
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
            best_w = w.copy()
        if num_inc == 0:
            break

        incorrect_x = X[inc_idxs][:1, :].transpose()
        incorrect_y = y[inc_idxs][0].squeeze()
        w += incorrect_y * incorrect_x

    return best_w


@timing
def pocket_with_lg(X, y, max_iter=-1):
    _, w = linear_regression(X, y)
    _, w = pocket_algorithm(X, y, max_iter=max_iter, w_init=w)
    return w


@timing
def linear_regression(X, y):
    X_t = X.transpose()
    return np.linalg.inv(X_t @ X) @ X_t @ y


@timing
def logistic_regression(X, y, max_iter, lr):
    X_t = X.transpose()

    def ce(w):
        N = X.shape[0]
        f = np.log(sigmoid(X @ w)).sum() / N
        gradient = X_t @ (y * sigmoid(-y * (X @ w)))

        return f, gradient

    w = gradient_ascent(ce, X.shape[1], (-inf, inf), max_steps=max_iter, lr=lr)
    return w


def get_perceptron_classifier():
    def perceptron(X, w):
        val = X @ w
        ones = np.ones(val.shape)
        return (val >= 0) * ones - (val < 0) * ones

    return perceptron


def get_logistic_regression_classifier(thr=0.5):
    def log_reg_class(X, w):
        val = sigmoid(X @ w)
        ones = np.ones(val.shape)
        return (val >= thr) * ones - (val < thr) * ones

    return log_reg_class
