from datetime import datetime
from math import sqrt, inf, log, e


_inf = 2e9

elem_wise = lambda *vectors, op: [op(*v) for v in zip(*vectors)]
add = lambda x, y: x + y
mult = lambda x, y: x * y
sub = lambda x, y: x - y
div = lambda x, y: x / y
inner_product = lambda x, y: sum([xi * yi for xi, yi in zip(x, y)])


def norm_data(data):
    d = len(data[0][0])
    N = len(data)

    # Computing empirical mean
    mean = [0] * d
    for x, _ in data:
        mean = elem_wise(x, mean, op=add)
    mean = elem_wise(mean, op=lambda x: x / N)

    # Computing empirical std
    var = [0] * d
    for x, _ in data:
        var = elem_wise(var, x, mean, op=lambda x, y, z: x + (y - z) ** 2)
    std = elem_wise(var, op=lambda x: sqrt(x / N))

    n_data = [
        (elem_wise(x, mean, std, op=lambda _val, _mean, _std: (_val - _mean) / _std), y)
        for x, y in data
    ]
    return n_data, mean, std


def unorm_weights(weights, mean, std):
    w_star, bias_star = weights[1:], weights[0]
    w = elem_wise(w_star, std, op=div)
    bias = bias_star - inner_product(w, mean)
    return [bias] + w


def gradient_descent(target, bounds, max_steps=10000, max_time=2):
    time_start = datetime.now()

    def check_bounds(w, lower=None, upper=None):
        lower = lower or [-1] * len(w)
        upper = upper or [1] * len(w)
        return all([l <= wi for l, wi in zip(lower, w)]) and all(
            [wi <= u for u, wi in zip(upper, w)]
        )

    best_f = inf
    best_w = None

    lower, upper = bounds
    d = len(lower)

    lr = 0.1
    eps = 1e-8
    beta = 0.9
    f_prev = inf
    w_prev = None

    w = [0 for u, l in zip(upper, lower)]
    gradient = [0] * d
    v = [0] * d

    it = 0
    while it < max_steps and (datetime.now() - time_start).seconds < max_time:
        it += 1

        f, gradient = target(w)

        if f < best_f:
            best_w = w
            best_f = f

        # Shrink lr if overstepped
        if f > f_prev and it > 10:
            lr /= 10
            w = w_prev

        # Updating v

        v = elem_wise(
            v, gradient, op=lambda x, y: sqrt(beta * x + (1 - beta) * (y ** 2)) + eps
        )
        checked = False
        # Adjusting learning rate if stepped out of the bounds
        for i in range(10):
            w = elem_wise(w, gradient, v, op=lambda x, y, z: x - lr * y / z)

            if check_bounds(w, lower, upper):
                checked = True
                break
            lr /= 10

        # If never got within bounds or change is less than epsilon, finish up
        if not checked or abs(f_prev - f) < eps:
            break
        else:
            f_prev = f
            w_prev = w

    return best_w, best_f, it, lr


def logistic_regression(data):

    N = len(data)
    d = len(data[0][0])

    def linear(w, x):
        w, bias = w[1:], w[0]
        return inner_product(w, x) + bias

    def ce(weights):
        f = 0
        gradient = [0] * (d + 1)

        for x, y in data:
            arg = y * linear(weights, x)
            f += log(1 + e ** (-arg))

            gradient = elem_wise(
                gradient, [1] + x, op=lambda a, b: a + (-y / (1 + e ** arg) * b)
            )

        return f / N, [g / N for g in gradient]

    w, _, _, _ = gradient_descent(
        ce, bounds=([-_inf] * (d + 1), [_inf] * (d + 1)), max_steps=100000, max_time=1,
    )

    return w


n, d = list(map(int, input().split()))
data = []
for _ in range(n):
    line = list(map(float, input().split()))
    data.append((line[:-1], line[-1]))
data, mean, std = norm_data(data)
w = logistic_regression(data)
w = unorm_weights(w, mean, std)
print(" ".join([str(x) for x in w]))
