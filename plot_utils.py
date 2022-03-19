import matplotlib.pyplot as plt
import numpy as np


def plot_line(w, x1, x2, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    plt.plot([x1, x2], [y1, y2], color=color, label=label)


def plot_error(ax, results, color, label, e_in=True, linestyle=None):
    X = results.keys()
    if e_in:
        y = [
            results[key][0][0] for key in X
        ]  # 0 is the index of the E_in results. 0 to take the mean
    else:
        y = [
            results[key][1][0] for key in X
        ]  # 1 is the index of the E_out results. 0 to take the mean

    ax.plot(X, y, color=color, label=f"{label}", linestyle=linestyle)


def plot_error_constant(ax, results, color, label, e_in=True, linestyle=None):
    if e_in:
        y = results[0][0]
    else:
        y = results[1][0]
    ax.axhline(y=y, color=color, label=label, linestyle=linestyle)


def plot_error_bars(ax, results, color, label, move=-1, e_in=True):
    labels = results.keys()
    X = np.arange(len(labels))  # the label locations
    if e_in:
        y = [
            results[key][0][0] for key in results.keys()
        ]  # 0 is the index of the E_in results. 0 to take the mean
    else:
        y = [
            results[key][1][0] for key in X
        ]  # 1 is the index of the E_out results. 0 to take the mean

    width = 0.25
    ax.bar(X + move * width / 2, y, width, label=label, color=color)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)


def plot_running_time(ax, results, color, label, linestyle=None):
    X = results.keys()
    y = [
        results[key][2][0] for key in X
    ]  # 2 is the index of the running time results. 0 to take the mean
    ax.plot(X, y, color=color, label=label, linestyle=linestyle)


def plot_running_time_constant(ax, results, color, label, linestyle=None):
    ax.axhline(y=results[2][0], color=color, label=label, linestyle=linestyle)


def plot_running_time_bars(ax, results, color, label, move=-1):
    labels = results.keys()
    X = np.arange(len(labels))  # the label locations
    y_in = [
        results[key][2][0] for key in results.keys()
    ]  # 2 is the index of the running time results. 0 to take the mean

    width = 0.25
    ax.bar(X + move * width / 2, y_in, width, label=label, color=color)
    ax.set_xticks(X)
    ax.set_xticklabels(labels)


def plot_pocket_lin_reg_results(p_results, plr_results, lr_results):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Plotting iterations vs error
    ax = axis[0]
    ax.set_xlabel("iterations")
    ax.set_ylabel("error")
    plot_error(ax, p_results, "orange", "w=0 init")
    plot_error(ax, p_results, "orange", "", e_in=False, linestyle="--")
    plot_error(ax, plr_results, "blue", "lin reg init")
    plot_error(ax, plr_results, "blue", "", e_in=False, linestyle="--")
    plot_error_constant(ax, lr_results, "red", "linear regression")
    plot_error_constant(ax, lr_results, "red", "", e_in=False, linestyle="--")
    ax.legend()

    # Plotting iterations vs running time
    ax = axis[1]
    ax.set_xlabel("iterations")
    ax.set_ylabel("running time (seconds)")
    plot_running_time(ax, p_results, "orange", "w_0 init")
    plot_running_time(ax, plr_results, "blue", "lin reg init")
    plot_running_time_constant(ax, lr_results, "red", "linear regression")
    ax.legend()

    plt.show()


def plot_normal_vs_outliers(p_results, plr_results, lr_results):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

    p_results, p_results_outliers = p_results
    plr_results, plr_results_outliers = plr_results
    lr_results, lr_results_outliers = lr_results

    # Plotting iterations vs error
    ax = axis[0]
    ax.set_xlabel("iterations")
    ax.set_ylabel("error")
    plot_error(ax, p_results, "orange", "w=0 init", e_in=False)
    plot_error(ax, p_results_outliers, "orange", "", e_in=False, linestyle="--")
    plot_error(ax, plr_results, "blue", "lin reg init", e_in=False)
    plot_error(ax, plr_results_outliers, "blue", "", e_in=False, linestyle="--")
    plot_error_constant(ax, lr_results, "red", "linear regression", e_in=False)
    plot_error_constant(ax, lr_results_outliers, "red", "", e_in=False, linestyle="--")
    ax.legend()

    # Plotting iterations vs running time
    ax = axis[1]
    ax.set_xlabel("iterations")
    ax.set_ylabel("running time (seconds)")
    plot_running_time(ax, p_results, "orange", "w_0 init")
    plot_running_time(ax, p_results_outliers, "orange", "", linestyle="--")
    plot_running_time(ax, plr_results, "blue", "lin reg init")
    plot_running_time(ax, plr_results_outliers, "blue", "", linestyle="--")
    plot_running_time_constant(ax, lr_results, "red", "linear regression")
    plot_running_time_constant(ax, lr_results_outliers, "red", "", linestyle="--")
    ax.legend()

    plt.show()


def plot_log_reg_results(
    log_reg_results,
    log_reg_results_outliers,
    log_reg_lr_results_separable,
    xlabel,
    bars=False,
):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axis[0]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    if not bars:
        plot_error(ax, log_reg_results, "orange", "w/o outliers", e_in=True)
        plot_error(ax, log_reg_results_outliers, "blue", "with outliers", e_in=True)
        plot_error(ax, log_reg_lr_results_separable, "red", "separable", e_in=True)
    else:
        plot_error_bars(
            ax, log_reg_results, "orange", "w/o outliers", move=-2, e_in=True
        )
        plot_error_bars(
            ax, log_reg_results_outliers, "blue", "with outliers", move=0, e_in=True
        )
        plot_error_bars(
            ax, log_reg_lr_results_separable, "red", "separable", move=2, e_in=True
        )
    ax.legend()

    ax = axis[1]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("running time (seconds)")
    if not bars:
        plot_running_time(ax, log_reg_results, "orange", "w/o outliers")
        plot_running_time(ax, log_reg_results_outliers, "blue", "with outliers")
        plot_running_time(ax, log_reg_lr_results_separable, "red", "separable")
    else:
        plot_running_time_bars(ax, log_reg_results, "orange", "w/o outliers", move=-2)
        plot_running_time_bars(
            ax, log_reg_results_outliers, "blue", "with outliers", move=0,
        )
        plot_running_time_bars(
            ax, log_reg_lr_results_separable, "red", "separable", move=2,
        )
    ax.legend()

    plt.show()

