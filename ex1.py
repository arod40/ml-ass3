import numpy as np
from tqdm import tqdm

from plot_utils import *
from models import *


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


if __name__ == "__main__":
    lg_exp = False
    pocket_and_lin_reg_exp = True
    log_reg_max_iter_exp = False
    log_reg_lr_exp = False

    np.random.seed(0)
    no_exps = 100

    datasets = []
    datasets_outliers = []
    for _ in range(no_exps):
        c = 3 * np.random.randn(2, 2)  # use the same center for both datasets
        datasets.append((get_data(c, outliers=False), get_data(c, outliers=False)))
        datasets_outliers.append(
            (get_data(c, outliers=True), get_data(c, outliers=False))
        )

    if pocket_and_lin_reg_exp:
        print("-----------------------------------------------------------")
        print("Running linear regression experiments")
        print("-----------------------------------------------------------")

        # Without outliers
        desc, (learning_alg, args, kwargs), classifier = get_experiment_setup("lin_reg")
        lr_results = experiment(
            desc, datasets, learning_alg, classifier, *args, **kwargs
        )
        for metric, (mean, std) in zip(["E_in", "E_out", "time"], lr_results):
            print(metric, "mean:", mean, "std:", std)

        # With outliers
        lr_results_outliers = experiment(
            desc + "[OUTLIERS]",
            datasets_outliers,
            learning_alg,
            classifier,
            *args,
            **kwargs,
        )
        for metric, (mean, std) in zip(["E_in", "E_out", "time"], lr_results_outliers):
            print(metric, "mean:", mean, "std:", std)

        print("-----------------------------------------------------------")
        print(
            "Running pocket algorithm variants experiments (varying max_iter to find out optimal values)"
        )
        print("-----------------------------------------------------------")

        pocket_max_iter_values = list(range(20, 1200, 20))
        p_results = {}
        plr_results = {}
        p_results_outliers = {}
        plr_results_outliers = {}
        for max_iter in pocket_max_iter_values:
            print("max_iter=", max_iter)
            # Pocket algorithm
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup("pa")
            kwargs["max_iter"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            p_results[max_iter] = metrics
            metrics = experiment(
                desc + "[OUTLIERS]",
                datasets_outliers,
                learning_alg,
                classifier,
                *args,
                **kwargs,
            )
            p_results_outliers[max_iter] = metrics

            # Pocket algorithm with linear regression
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup(
                "pa_lin_reg"
            )
            kwargs["max_iter"] = max_iter
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            plr_results[max_iter] = metrics
            metrics = experiment(
                desc + "[OUTLIERS]",
                datasets_outliers,
                learning_alg,
                classifier,
                *args,
                **kwargs,
            )
            plr_results_outliers[max_iter] = metrics

        # plot_pocket_lin_reg_results(p_results, plr_results, lr_results)
        plot_normal_vs_outliers(
            (p_results, p_results_outliers),
            (plr_results, plr_results_outliers),
            (lr_results, lr_results_outliers),
        )

    if log_reg_max_iter_exp:
        print("-----------------------------------------------------------")
        print(
            "(Running logistic regression with different maximum number of iterations"
        )
        print("-----------------------------------------------------------")

        log_reg_max_iter_values = list(range(1, 201, 10))
        log_reg_max_iter_results = {}
        log_reg_max_iter_results_outliers = {}
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

            metrics = experiment(
                desc + "[OUTLIERS]",
                datasets_outliers,
                learning_alg,
                classifier,
                *args,
                **kwargs,
            )
            for metric, (mean, std) in zip(["E_in", "E_out", "time"], metrics):
                print(metric, "mean:", mean, "std:", std)
            log_reg_max_iter_results_outliers[max_iter] = metrics

        plot_log_reg_results(
            log_reg_max_iter_results,
            log_reg_max_iter_results_outliers,
            xlabel="iterations",
        )

    if log_reg_lr_exp:
        print("-----------------------------------------------------------")
        print("(Running logistic regression with different learning rates")
        print("-----------------------------------------------------------")

        lr_values = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        log_reg_lr_results = {}
        log_reg_lr_results_outliers = {}
        for lr in lr_values:
            desc, (learning_alg, args, kwargs), classifier = get_experiment_setup(
                "log_reg"
            )
            print("lr=", lr)
            kwargs["lr"] = lr
            kwargs["max_iter"] = 400
            metrics = experiment(
                desc, datasets, learning_alg, classifier, *args, **kwargs
            )
            for metric, (mean, std) in zip(["E_in", "E_out", "time"], metrics):
                print(metric, "mean:", mean, "std:", std)
            log_reg_lr_results[lr] = metrics

            metrics = experiment(
                desc + "[OUTLIERS]",
                datasets_outliers,
                learning_alg,
                classifier,
                *args,
                **kwargs,
            )
            for metric, (mean, std) in zip(["E_in", "E_out", "time"], metrics):
                print(metric, "mean:", mean, "std:", std)
            log_reg_lr_results_outliers[lr] = metrics

        plot_log_reg_results(
            log_reg_lr_results,
            log_reg_lr_results_outliers,
            xlabel="lr",
            bars=True,
            e_in=True,
            e_out=False,
        )

