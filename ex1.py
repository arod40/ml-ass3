import sys
from math import inf

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


sigmoid = np.vectorize(lambda x: 1 / (1 + np.e ** (-x)))


def gradient_ascent(target, d, bounds, max_steps=10000, lr=0.1):
    lower, upper = bounds

    best_f = inf
    best_w = None
    opt_it = None
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
            opt_it = it

    return best_w, opt_it


def pocket_algorithm(X, y, max_iter=-1, w_init=None):
    def perceptron(x, w):
        val = x @ w
        ones = np.ones(val.shape)
        return (val >= 0) * ones - (val < 0) * ones

    w = w_init if w_init is not None else np.zeros((X.shape[1], 1))
    least_inc = inf
    best_w = None
    opt_it = None
    it = 0
    while max_iter == -1 or it < max_iter:
        it += 1
        inc_idxs = (perceptron(X, w) != y).squeeze()
        num_inc = inc_idxs.sum()

        if num_inc < least_inc:
            least_inc = num_inc
            best_w = w.copy()
            opt_it = it
        if num_inc == 0:
            break

        incorrect_x = X[inc_idxs][:1, :].transpose()
        incorrect_y = y[inc_idxs][0].squeeze()
        w += incorrect_y * incorrect_x

    return best_w, opt_it


def linear_regression(X, y):
    X_t = X.transpose()
    return np.linalg.inv(X_t @ X) @ X_t @ y


def logistic_regression(X, y, max_iter, lr):
    X_t = X.transpose()

    def ce(w):
        N = X.shape[0]
        f = np.log(sigmoid(X @ w)).sum() / N
        gradient = X_t @ (y * sigmoid(-y * (X @ w)))

        return f, gradient

    w, opt_it, = gradient_ascent(ce, X.shape[1], (-inf, inf), max_steps=max_iter, lr=lr)
    return w, opt_it


def plot_line(w, x1, x2, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    plt.plot([x1, x2], [y1, y2], color=color, label=label)


def get_data(c):
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


def get_perceptron_classifier(w):
    def perceptron(X):
        val = X @ w
        ones = np.ones(val.shape)
        return (val >= 0) * ones - (val < 0) * ones

    return perceptron


def get_logistic_regression_classifier(w, prob=0.5):
    def log_reg_class(X):
        val = sigmoid(X @ w)
        ones = np.ones(val.shape)
        return (val >= prob) * ones - (val < prob) * ones

    return log_reg_class


def evaluate(class_func, X, y):
    return (class_func(X) != y).sum() / X.shape[0]


if __name__ == "__main__":
    np.random.seed(0)

    item = sys.argv[1]

    no_exps = 100
    c = 3 * np.random.randn(2, 2)  # use the same center for both datasets
    datasets = [(get_data(c), get_data(c)) for _ in range(no_exps)]

    if item == "1":
        key = "---E_out: {}\n--- iterations: {}\n"

        p_error = 0
        p_it = 0
        lin_reg_error = 0
        p2_error = 0
        p2_it = 0
        log_reg_inerror = 0
        log_reg_outerror = 0
        log_reg_it = 0

        p_error_outliers = 0
        p_it_outliers = 0
        lin_reg_error_outliers = 0
        p2_error_outliers = 0
        p2_it_outliers = 0
        log_reg_inerror_outliers = 0
        log_reg_outerror_outliers = 0
        log_reg_it_outliers = 0

        def eval(X_in, y_in, X_out, y_out):
            w_p, it_p = pocket_algorithm(X_in, y_in, max_iter=10000)
            w_lg = linear_regression(X_in, y_in)
            w_p2, it_p2 = pocket_algorithm(X_in, y_in, max_iter=10000, w_init=w_lg)
            w_log_reg, it_log_reg = logistic_regression(
                X_in, y_in, max_iter=10000, lr=0.1
            )

            p_class = get_perceptron_classifier(w_p)
            lg_class = get_perceptron_classifier(w_lg)
            p2_class = get_perceptron_classifier(w_p2)
            log_reg_class = get_logistic_regression_classifier(w_log_reg)

            p_error = evaluate(p_class, X_out, y_out)
            p_it = it_p
            lin_reg_error = evaluate(lg_class, X_out, y_out)
            p2_error = evaluate(p2_class, X_out, y_out)
            p2_it = it_p2
            log_reg_inerror = evaluate(log_reg_class, X_in, y_in)
            log_reg_outerror = evaluate(log_reg_class, X_out, y_out)
            log_reg_it = it_log_reg

            return (
                p_error,
                p_it,
                lin_reg_error,
                p2_error,
                p2_it,
                log_reg_inerror,
                log_reg_outerror,
                log_reg_it,
            )

        for (X_in, y_in), (X_out, y_out) in tqdm(datasets):

            # Without outliers
            (
                _p_error,
                _p_it,
                _lin_reg_error,
                _p2_error,
                _p2_it,
                _log_reg_inerror,
                _log_reg_outerror,
                _log_reg_it,
            ) = eval(X_in, y_in, X_out, y_out)
            p_error += _p_error
            p_it += _p_it
            lin_reg_error += _lin_reg_error
            p2_error += _p2_error
            p2_it += _p2_it
            log_reg_inerror += _log_reg_inerror
            log_reg_outerror += _log_reg_outerror
            log_reg_it += _log_reg_it

            # With outliers
            y_outliers = add_outliers(y_in)
            (
                _p_error,
                _p_it,
                _lin_reg_error,
                _p2_error,
                _p2_it,
                _log_reg_inerror,
                _log_reg_outerror,
                _log_reg_it,
            ) = eval(X_in, y_outliers, X_out, y_out)
            p_error_outliers += _p_error
            p_it_outliers += _p_it
            lin_reg_error_outliers += _lin_reg_error
            p2_error_outliers += _p2_error
            p2_it_outliers += _p2_it
            log_reg_inerror_outliers += _log_reg_inerror
            log_reg_outerror_outliers += _log_reg_outerror
            log_reg_it_outliers += _log_reg_it

        print("Pocket Algorithm")
        print(key.format(p_error / no_exps, p_it / no_exps))
        print("Linear Regression")
        print(key.format(lin_reg_error / no_exps, "(invalid)"))
        print("Pocket Algorithm w LG")
        print(key.format(p2_error / no_exps, p2_it / no_exps))
        print("Logistic Regression")
        print(key.format(log_reg_outerror / no_exps, log_reg_it / no_exps))

        print("[OUTLIERS] Pocket Algorithm")
        print(key.format(p_error_outliers / no_exps, p_it_outliers / no_exps))
        print("[OUTLIERS] Linear Regression")
        print(key.format(lin_reg_error / no_exps, "(invalid)"))
        print("[OUTLIERS] Pocket Algorithm w LG")
        print(key.format(p2_error_outliers / no_exps, p2_it_outliers / no_exps))
        print("[OUTLIERS] Logistic Regression")
        print(
            key.format(
                log_reg_outerror_outliers / no_exps, log_reg_it_outliers / no_exps
            )
        )

    if item == "2":
        key = "---E_in: {}\n---E_out: {}\n--- iterations: {}\n"

        def eval(
            learning_alg,
            get_classifier_fn,
            X_in,
            y_in,
            X_out,
            y_out,
            return_it=True,
            *args,
            **kwargs,
        ):
            if return_it:
                w, it = learning_alg(X_in, y_in, *args, **kwargs)
            else:
                w = learning_alg(X_in, y_in, *args, **kwargs)

            classifier = get_classifier_fn(w)

            in_error = evaluate(classifier, X_in, y_in)
            out_error = evaluate(classifier, X_out, y_out)

            return in_error, out_error, it

        max_iter = 10000
        lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
        for lr in lrs:
            print(f"learning rate: {lr}")
            in_error = 0
            out_error = 0
            it = 0

            in_error_outliers = 0
            out_error_outliers = 0
            it_outliers = 0
            for (X_in, y_in), (X_out, y_out) in tqdm(datasets):
                _in_error, _out_error, _it = eval(
                    logistic_regression,
                    get_logistic_regression_classifier,
                    X_in,
                    y_in,
                    X_out,
                    y_out,
                    return_it=True,
                    max_iter=max_iter,
                    lr=lr,
                )
                in_error += _in_error
                out_error += _out_error
                it += _it

                # With outliers
                y_outliers = add_outliers(y_in)
                _in_error, _out_error, _it = eval(
                    logistic_regression,
                    get_logistic_regression_classifier,
                    X_in,
                    y_outliers,
                    X_out,
                    y_out,
                    return_it=True,
                    max_iter=max_iter,
                    lr=lr,
                )
                in_error_outliers += _in_error
                out_error_outliers += _out_error
                it_outliers += _it

            print(f"Logistic Regression")
            print(key.format(in_error / no_exps, out_error / no_exps, it / no_exps))
            print(f"[OUTLIERS] Logistic Regression")
            print(
                key.format(
                    in_error_outliers / no_exps,
                    out_error_outliers / no_exps,
                    it_outliers / no_exps,
                )
            )

    if item == "show":
        np.random.seed(9)

        c = 3 * np.random.randn(2, 2)  # use the same center for both datasets
        X, y = get_data(c)
        y_ = y.squeeze()

        w_p, it_p = pocket_algorithm(X, y, max_iter=10000)
        w_lg = linear_regression(X, y)
        w_p2, it_p2 = pocket_algorithm(X, y, max_iter=10000, w_init=w_lg)
        w_log_reg, it_log_reg = logistic_regression(X, y, max_iter=10000, lr=0.1)
        w_log_reg, _ = logistic_regression(X, y, max_iter=10000, lr=0.1)

        p_class = get_perceptron_classifier(w_p)
        log_reg_class = get_logistic_regression_classifier(w_log_reg)
        p_error = evaluate(p_class, X, y)
        log_reg_error = evaluate(log_reg_class, X, y)
        print(p_error, log_reg_error)

        plt.xlim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.ylim(X[:, 2].min() - 1, X[:, 2].max() + 1)

        plt.scatter(X[y_ == 1, 1], X[y_ == 1, 2], marker="o", color="blue")
        plt.scatter(X[y_ == -1, 1], X[y_ == -1, 2], marker="x", color="red")

        plot_line(
            list(w_p), X[:, 1].min(), X[:, 1].max(), "red", label="pocket algorithm",
        )
        plot_line(
            list(w_lg), X[:, 1].min(), X[:, 1].max(), "blue", label="linear regression",
        )
        plot_line(
            list(w_p2),
            X[:, 1].min(),
            X[:, 1].max(),
            "green",
            label="pocket with lin regression init",
        )
        plot_line(
            list(w_log_reg),
            X[:, 1].min(),
            X[:, 1].max(),
            "orange",
            label="logistic regression",
        )
        plt.legend()
        plt.show()

