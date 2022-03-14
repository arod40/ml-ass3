import numpy as np
import matplotlib.pyplot as plt


def get_data():
    c = 3 * np.random.randn(2, 2)

    a = np.array([1, 2, 3])
    a.squeeze

    y = np.concatenate([-1 * np.ones((50)), np.ones((50))])
    x = np.random.randn(100, 2) + c[((y + 3) / 2 - 1).astype("int64")]
    x = np.concatenate([np.ones((100, 1)), x], axis=1)

    return x, y


if __name__ == "__main__":
    x, y = get_data()

    plt.scatter(x[y == 1, 1], x[y == 1, 2], marker="o")
    plt.scatter(x[y == -1, 1], x[y == -1, 2], marker="x")

    plt.show()

