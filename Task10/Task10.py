import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def cholesky_tridiagonal(diagonal: matrix, subdiagonal: matrix) -> tuple[matrix, matrix]:
    n = diagonal.shape[0]
    C_diag = np.zeros(n, dtype=np.float64)
    C_subdiag = np.zeros(n - 1, dtype=np.float64)

    C_diag[0] = np.sqrt(diagonal[0])
    for i in range(1, n):
        C_subdiag[i - 1] = subdiagonal[i - 1] / C_diag[i - 1]
        C_diag[i] = np.sqrt(diagonal[i] - C_subdiag[i - 1] ** 2)

    return C_diag, C_subdiag


def forward_substitution_tridiagonal(diagonal: matrix, subdiagonal: matrix, b: matrix) -> vector:
    n = b.shape[0]
    y = np.zeros(n, dtype=np.float64)

    y[0] = b[0] / diagonal[0]
    for i in range(1, n):
        y[i] = (b[i] - subdiagonal[i - 1] * y[i - 1]) / diagonal[i]

    return y


def backward_substitution_tridiagonal(diagonal: matrix, subdiagonal: matrix, y: matrix) -> vector:
    n = y.shape[0]
    z = np.zeros(n, dtype=np.float64)

    z[-1] = y[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        z[i] = (y[i] - subdiagonal[i] * z[i + 1]) / diagonal[i]

    return z

def get_spline_coefficients(x_val: num, x_j: num, h: num) -> tuple[num, num, num, num]:
    # x_{j+1} - x{j} = h
    # x_{j+1} = h + x{j}

    A = (x_j + h - x_val) / h
    B = (x_val - x_j) / h

    h_squared = h * h

    # x_{j+1} - x{j} = h
    C = (1.0 / 6.0) * (A ** 3 - A) * h_squared
    D = (1.0 / 6.0) * (B ** 3 - B) * h_squared

    return A, B, C, D

def spline_function():
    pass

def f_x(x: num) -> num:
    return 1.0 / (1 + 5 * x * x)

def main():
    x: vector = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype=num)
    n = len(x)
    f: vector = np.zeros(n, dtype=num)

    for i in range(n):
        f[i] = f_x(x[i])

    diagonal: vector = np.full(n - 2, 4.0)
    subdiagonal: vector = np.full(n - 3, 1.0)

    # stałe h
    h = x[1] - x[0]
    b: vector = np.zeros(n - 2, dtype=num)
    for i in range(n - 2):
        b[i] = (6.0 / (h * h) * (f[i + 2] - 2*f[i + 1] + f[i]))

    C_diag, C_subdiag = cholesky_tridiagonal(diagonal, subdiagonal)
    y = forward_substitution_tridiagonal(C_diag, C_subdiag, b)
    ksi_without_zero = backward_substitution_tridiagonal(C_diag, C_subdiag, y)

    ksi = np.zeros(n, dtype=num)
    ksi[1:n-1] = ksi_without_zero



if __name__ == "__main__":
    main()