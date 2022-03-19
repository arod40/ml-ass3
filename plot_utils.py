import matplotlib.pyplot as plt


def plot_line(w, x1, x2, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    plt.plot([x1, x2], [y1, y2], color=color, label=label)


def plot_error(ax, results, color, label, constant=False, bars=False, move=-1):
    if not constant and not bars:
        X = results.keys()
        y_in = [
            results[key][0][0] for key in X
        ]  # 0 is the index of the E_in results. 0 to take the mean
        y_out = [
            results[key][1][0] for key in X
        ]  # 1 is the index of the E_out results. 0 to take the mean
        ax.plot(X, y_in, color=color, linestyle="--")
        ax.plot(X, y_out, color=color, label=f"{label}")
    elif not bars:
        ax.axhline(y=results[0][0], color=color, linestyle="--")
        ax.axhline(y=results[1][0], color=color, label=label)
    else:
        labels = results.keys()
        X = np.arange(len(labels))  # the label locations
        y_in = [
            results[key][0][0] for key in results.keys()
        ]  # 0 is the index of the E_in results. 0 to take the mean

        width = 0.35
        ax.bar(X + move * width / 2, y_in, width, label=label, color=color)
        ax.set_xticks(X)
        ax.set_xticklabels(labels)


def plot_running_time(ax, results, color, label, constant=False, bars=False, move=-1):
    if not constant and not bars:
        X = results.keys()
        y = [
            results[key][2][0] for key in X
        ]  # 2 is the index of the running time results. 0 to take the mean
        ax.plot(X, y, color=color, label=label)
    elif not bars:
        ax.axhline(y=results[2][0], color=color, label=label)
    else:
        labels = results.keys()
        X = np.arange(len(labels))  # the label locations
        y_in = [
            results[key][2][0] for key in results.keys()
        ]  # 2 is the index of the running time results. 0 to take the mean

        width = 0.35
        ax.bar(X + move * width / 2, y_in, width, label=label, color=color)
        ax.set_xticks(X)
        ax.set_xticklabels(labels)


def plot_pocket_results(p_results, plr_results, lr_results=None):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

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


def plot_log_reg_results(log_reg_results, log_reg_results_outliers, xlabel, bars=False):
    figure, axis = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axis[0]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    plot_error(ax, log_reg_results, "orange", "w/o outliers", bars=bars, move=-1)
    plot_error(ax, log_reg_results_outliers, "blue", "with outliers", bars=bars, move=1)
    ax.legend()

    ax = axis[1]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("running time (seconds)")
    plot_running_time(ax, log_reg_results, "orange", "w/o outliers", bars=bars, move=-1)
    plot_running_time(
        ax, log_reg_results_outliers, "blue", "with outliers", bars=bars, move=1,
    )
    ax.legend()

    plt.show()

