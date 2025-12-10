import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

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
            sign = (-1) ** i
            prod = 1.0
            for j in range(i, i + d + 1):
                if j != k:
                    prod *= 1.0 / (x[k] - x[j])
            s += sign * prod
        w[k] = s
    return w

def main():
    x: vector = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype=num)
    n = len(x)
    f: vector = np.zeros(n, dtype=num)

    for i in range(n):
        f[i] = f_x(x[i])

if __name__ == "__main__":
    main()
