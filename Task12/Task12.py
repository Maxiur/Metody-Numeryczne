import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.special import comb

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def f_x(x: num) -> num:
    return 1.0 / (1 + 5 * x * x)

def floater_hormann_weights(x: vector, d: int) -> vector:
    n = len(x)
    w = np.zeros(n, dtype=num)

    for k in range(n):
        start = max(0, k - d)
        end = min(k, n - d - 1)

        s = 0.0
        for i in range(start, end + 1):
            s += comb(d, k - i, exact=True)

        sign = 1 if (k - d) % 2 == 0 else -1
        w[k] = sign * s

    return w

def floater_hormann_function(x_nodes: vector, f:vector, w: vector, x_eval: vector) -> vector:
    r = np.zeros_like(x_eval)
    for idx, x in enumerate(x_eval):
        diff = x - x_nodes

        mask = np.isclose(diff, 0.0, atol=1e-12)
        if np.any(mask):
            r[idx] = f[mask][0]
        else:
            # w_diff = w / (x - x_k)
            w_diff = w / diff
            numerator = np.sum(w_diff * f)
            denominator = np.sum(w_diff)
            r[idx] = numerator / denominator

    return r


def main():
    x: vector = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype=num)
    n = len(x)
    f: vector = np.zeros(n, dtype=num)

    for i in range(n):
        f[i] = f_x(x[i])

    d = 3
    w = floater_hormann_weights(x, d)
    print(w)

    x_eval = np.linspace(-1.25, 1.25, 400)
    y_eval = floater_hormann_function(x, f, w, x_eval)

    plt.plot(x_eval, y_eval, label="Interpolacja Floatera–Hormanna (d=3)")
    plt.plot(x, f, 'ro', label="Węzły")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
