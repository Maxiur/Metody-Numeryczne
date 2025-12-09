import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def thomas_factor(subdiagonal: vector, diagonal: vector) -> tuple[vector, vector]:
    n = len(diagonal)
    L = np.zeros(n - 1)

    for i in range(1, n):
        L[i - 1] = subdiagonal[i - 1] / diagonal[i - 1]
        diagonal[i] -= L[i - 1] * subdiagonal[i - 1]

    return L, diagonal

def thomas_solve(L: vector, U: vector, b: vector, superdiagonal: vector) -> vector:
    n = len(U)

    # forward
    for i in range(1, n):
        b[i] -= L[i - 1] * b[i - 1]

    # back
    x = np.zeros(n)
    x[-1] = b[-1] / U[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - superdiagonal[i] * x[i + 1]) / U[i]

    return x

def get_spline_coefficients(x_val: num, x_j: num, x_j1: num) -> tuple[num, num, num, num]:
    h = x_j1 - x_j

    A = (x_j1 - x_val) / h
    B = (x_val - x_j) / h

    h_squared = h * h

    # x_{j+1} - x{j} = h
    C = (1.0 / 6.0) * (A ** 3 - A) * h_squared
    D = (1.0 / 6.0) * (B ** 3 - B) * h_squared

    return A, B, C, D

def spline_function(x_val: num, x: vector, f: vector, ksi: vector) -> num:
    j = np.searchsorted(x, x_val) - 1
    j = max(0, min(j, len(x) - 2))

    A, B, C, D = get_spline_coefficients(x_val, x[j], x[j + 1])

    P_x = A * f[j] + B * f[j + 1] + C * ksi[j] + D * ksi[j + 1]
    return P_x


def plot_results(x: vector, f: vector, ksi: vector):
    # Generowanie punktów do rysowania splajnu
    x_range = np.linspace(-1.25, 1.25, 1000)

    # Obliczanie wartości splajnu w punktach x_range
    s_spline = np.zeros_like(x_range)
    for k, val in enumerate(x_range):
        s_spline[k] = spline_function(val, x, f, ksi)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, s_spline, label='Naturalny Splajn Kubiczny $P(x)$', color='blue')
    plt.plot(x, f, 'o', label='Węzły interpolacji', color='red')
    plt.title('Naturalny Splajn Kubiczny (węzły Czebyszewa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

def f_x(x: num) -> num:
    return 1.0 / (1 + 5 * x * x)

def x_j(j: int) -> num:
    return -np.cos(((2*j - 1) / 16)* np.pi)

def main():
    n = 8
    x: vector = np.zeros(n, dtype=num)
    for i in range(n):
        x[i] = x_j(i + 1)

    f: vector = np.zeros(n, dtype=num)
    for i in range(n):
        f[i] = f_x(x[i])

    # długość każdego segmentu h[i] = x[i+1] - x[i]
    h = np.diff(x)

    diagonal: vector = np.zeros(n - 2, dtype=num)
    subdiagonal: vector = np.zeros(n - 3, dtype=num)
    superdiagonal: vector = np.zeros(n - 3, dtype=num)

    b: vector = np.zeros(n - 2, dtype=num)

    for i in range(n - 3):
        diagonal[i] = 2 * (h[i] + h[i + 1])
        subdiagonal[i] = h[i + 1]
        superdiagonal[i] = h[i + 1]
        b[i] = 6 * ((f[i + 2] - f[i + 1]) / h[i + 1] - (f[i + 1] - f[i]) / h[i])

    diagonal[-1] = 2 * (h[-2] + h[-1])
    b[-1] = 6 * ((f[-1] - f[-2]) / h[-1] - (f[-2] - f[-3]) / h[-2])

    L, U = thomas_factor(subdiagonal, diagonal)
    ksi_without_zero = thomas_solve(L, U, b, superdiagonal)

    ksi = np.zeros(n, dtype=num)
    ksi[1:n-1] = ksi_without_zero

    plot_results(x, f, ksi)

if __name__ == "__main__":
    main()
