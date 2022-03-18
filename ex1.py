from cProfile import label
import sys
from math import inf
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def timing(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res

    return wrapper


sigmoid = np.vectorize(lambda x: 1 / (1 + np.e ** (-x)))


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


def get_data(c, num_data_points=100, outliers=False):
    assert num_data_points % 2 == 0
    y = np.concatenate(
        [-1 * np.ones((num_data_points // 2, 1)), np.ones((num_data_points // 2, 1))]
    )
    X = (
        np.random.randn(num_data_points, 2)
        + c[((y.squeeze() + 3) / 2 - 1).astype("int64")]
    )
    X = np.concatenate([np.ones((num_data_points, 1)), X], axis=1)

    if outliers:
        y = add_outliers(y)

    return X, y


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


def evaluate(class_func, w, X, y):
    return (class_func(X, w) != y).sum() / X.shape[0]


def experiment(
    name, datasets, learning_algorithm, classifier, plot_results=False, *args, **kwargs
):
    print(name)
    agg_eins = []
    agg_eouts = []
    agg_times = []
    for (X_in, y_in), (X_out, y_out) in tqdm(datasets):
        runtime, w = learning_algorithm(X_in, y_in, *args, **kwargs)
        agg_eins.append(evaluate(classifier, w, X_in, y_in))
        agg_eouts.append(evaluate(classifier, w, X_out, y_out))
        agg_times.append(runtime)

        if plot_results:
            X, y = X_in, y_in.squeeze()
            plt.title(f"{name} decision boundary and training data points")
            plt.xlim(X[:, 1].min() - 1, X[:, 1].max() + 1)
            plt.ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)

            plt.scatter(X[y == 1, 1], X[y == 1, 2], marker="o", color="blue")
            plt.scatter(X[y == -1, 1], X[y == -1, 2], marker="x", color="red")

            plot_line(
                list(w), X[:, 1].min(), X[:, 1].max(), "black",
            )
            print(np.mean(evaluate(classifier, w, X_in, y_in)))
            plt.show()

    return [(np.mean(vals), np.std(vals)) for vals in [agg_eins, agg_eouts, agg_times]]


def get_experiment_setup(alg_code):
    alg_code2desc = {
        "pa": "Pocket algorithm",
        "lin_reg": "Linear regression",
        "pa_lin_reg": "Pocket alg. with linear reg. initial weights",
        "log_reg": "Logistic regression",
    }
    learning_algs = {
        "pa": (pocket_algorithm, [], {"max_iter": 10000}),
        "lin_reg": (linear_regression, [], {}),
        "pa_lin_reg": (pocket_with_lg, [], {"max_iter": 10000}),
        "log_reg": (logistic_regression, [], {"max_iter": 10000, "lr": 0.1}),
    }

    perceptron_classifier = get_perceptron_classifier()
    logistic_regression_classifier = get_logistic_regression_classifier(thr=0.5)
    classifiers = {
        "pa": perceptron_classifier,
        "lin_reg": perceptron_classifier,
        "pa_lin_reg": perceptron_classifier,
        "log_reg": logistic_regression_classifier,
    }

    return alg_code2desc[alg_code], learning_algs[alg_code], classifiers[alg_code]


def plot_line(w, x1, x2, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    plt.plot([x1, x2], [y1, y2], color=color, label=label)


def plot_pocket_results(p_results, plr_results, lr_results=None):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

    def plot_error(ax, results, color, label, constant=False):
        ax.set_xlabel("iterations")
        ax.set_ylabel("error")
        if not constant:
            X = results.keys()
            y_in = [
                results[key][0][0] for key in X
            ]  # 1 is the index of the E_out results. 0 to take the mean
            y_out = [
                results[key][1][0] for key in X
            ]  # 1 is the index of the E_out results. 0 to take the mean
            ax.plot(X, y_in, color=color, linestyle="--")
            ax.plot(X, y_out, color=color, label=f"{label}")
        else:
            ax.axhline(y=results[0][0], color=color, linestyle="--")
            ax.axhline(y=results[1][0], color=color, label=label)

    def plot_running_time(ax, results, color, label, constant=False):
        if not constant:
            X = results.keys()
            y = [
                results[key][2][0] for key in X
            ]  # 2 is the index of the running time results. 0 to take the mean
            ax.plot(X, y, color=color, label=label)
        else:
            ax.axhline(y=results[2][0], color=color, label=label)

    # Plotting iterations vs error
    ax = axis[0]
    ax.set_xlabel("iterations")
    ax.set_ylabel("error")
    plot_error(ax, p_results, "orange", "w=0 init")
    plot_error(ax, plr_results, "blue", "lin reg init")
    if lr_results:
        plot_error(ax, lr_results, "red", "linear regression", constant=True)
    ax.legend()

    # Plotting iterations vs running time
    ax = axis[1]
    ax.set_xlabel("iterations")
    ax.set_ylabel("running time (seconds)")
    plot_running_time(ax, p_results, "orange", "w_0 init")
    plot_running_time(ax, plr_results, "blue", "lin reg init")
    if lr_results:
        plot_running_time(ax, lr_results, "red", "linear regression", constant=True)
    ax.legend()

    plt.show()


def plot_log_reg_results(log_reg_results):
    figure, axis = plt.subplots(1, 1, figsize=(6, 4, 4.8))

    # Plotting iterations vs error
    ax = axis[0]
    ax.set_xlabel("iterations")
    ax.set_ylabel("E_out")
    X = log_reg_results.keys()
    y = [
        log_reg_results[key][1][0] for key in X
    ]  # 1 is the index of the E_out results. 0 to take the mean
    ax.plot(X, y, color="orange")

    # Plotting iterations vs running time
    ax = axis[1]
    ax.set_xlabel("iterations")
    ax.set_ylabel("running time")
    X = log_reg_results.keys()
    y = [
        log_reg_results[key][2][0] for key in X
    ]  # 2 is the index of the running time results. 0 to take the mean
    ax.plot(X, y, color="orange")


if __name__ == "__main__":
    outliers = False
    lg_exp = True
    pocket_exp = True
    log_reg_max_iter_exp = False
    log_reg_lr_exp = False

    np.random.seed(0)
    no_exps = 1000

    datasets = []
    for _ in range(no_exps):
        c = 3 * np.random.randn(2, 2)  # use the same center for both datasets
        datasets.append(
            (get_data(c, outliers=outliers), get_data(c, outliers=outliers))
        )

    if lg_exp:
        print("-----------------------------------------------------------")
        print("Running linear regression experiments")
        print("-----------------------------------------------------------")

        desc, (learning_alg, args, kwargs), classifier = get_experiment_setup("lin_reg")
        lr_metrics = experiment(
            desc, datasets, learning_alg, classifier, *args, **kwargs
        )
        for metric, (mean, std) in zip(["E_in", "E_out", "time"], lr_metrics):
            print(metric, "mean:", mean, "std:", std)

    if pocket_exp:
        print("-----------------------------------------------------------")
        print(
            "Running pocket algorithm variants experiments (varying max_iter to find out optimal values)"
        )
        print("-----------------------------------------------------------")

        pocket_max_iter_values = list(range(20, 1200, 20))
        p_results = {}
        plr_results = {}
        for max_iter in pocket_max_iter_values:
            print("max_iter=", max_iter)
            # Pocket algorithm
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup("pa")
            kwargs["max_iter"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            p_results[max_iter] = metrics

            # Pocket algorithm with linear regression
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup(
                "pa_lin_reg"
            )
            kwargs["max_iter"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            plr_results[max_iter] = metrics

        plot_pocket_results(
            p_results, plr_results, lr_results=lr_metrics if lg_exp else None
        )

    if log_reg_max_iter_exp:
        print("-----------------------------------------------------------")
        print(
            "(Running logistic regression with different maximum number of iterations"
        )
        print("-----------------------------------------------------------")

        log_reg_max_iter_values = list(range(100, 1000, 50))
        log_reg_max_iter_results = {}
        for max_iter in log_reg_max_iter_values:
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup(
                "log_reg"
            )
            print("max_iter=", max_iter)
            kwargs["max_iter"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            for metric, (mean, std) in zip(["E_in", "E_out", "time"], metrics):
                print(metric, "mean:", mean, "std:", std)
            log_reg_max_iter_results[max_iter] = metrics

    if log_reg_lr_exp:
        print("-----------------------------------------------------------")
        print("(Running logistic regression with different learning rates")
        print("-----------------------------------------------------------")

        lr_values = [0.1, 0.05, 0.01, 0.005, 0.001]
        log_reg_lr_results = {}
        for lr in lr_values:
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup(
                "log_reg"
            )
            print("lr=", lr)
            kwargs["lr"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            for metric, (mean, std) in zip(["E_in", "E_out", "time"], metrics):
                print(metric, "mean:", mean, "std:", std)
            log_reg_lr_results[lr] = metrics
