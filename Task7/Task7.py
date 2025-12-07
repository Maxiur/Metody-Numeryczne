import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def l_j(x_val: num, x: vector, j: int) -> num:
    n = len(x)
    numerator: num = np.float64(1.0)
    denominator: num = np.float64(1.0)

    # 0...j-1
    for k in range(0, j):
        numerator *= x_val - x[k]
        denominator *= x[j] - x[k]

    # j+1...n-1
    for k in range(j + 1, n):
        numerator *= x_val - x[k]
        denominator *= x[j] - x[k]

    return numerator / denominator

def coefficients(x: vector, f: vector) -> matrix:
    n = len(x)

    lj = np.zeros(n, dtype=num)
    for j in range(n):
        lj[j] = l_j(0, x, j)

    # współczynniki
    a = np.zeros(n, dtype=num)
    f_k = f.copy()

    for k in range(n):
        a[k] = np.sum(lj * f_k)

        for j in range(n):
            f_k[j] = (f_k[j] - a[k]) / x[j]

    return a

def P(x: num, a: vector) -> num:
    # Wyliczenie wielomianu schematem hornera
    val = 0.0
    for coeff in reversed(a):
        val = val * x + coeff

    return val

def main():
    x: vector = np.array([-1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00], dtype=num)
    f: vector = np.array([
        6.00000000000000,
        3.04034423828125,
        1.74218750000000,
        1.26361083984375,
        0.75982666015625,
        0.63281250000000,
        0.85809326171875,
        2.00000000000000
    ], dtype=num)

    coeff = coefficients(x, f)
    print(coeff)

    x_plot = np.linspace(-2, 1.25, 1000)
    y_plot = np.array([P(x, coeff) for x in x_plot])

    plt.plot(x_plot, y_plot, label="Wielomian interpolacyjny", color="blue")

    plt.scatter(x, f, color="red", label="Punkty interpolacji", zorder=5)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Interpolacja Lagrange'a")
    plt.grid(True)
    plt.legend()
    plt.xlim(-2, 1.25)
    plt.show()

if __name__ == "__main__":
    main()
