import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
num = np.float64

def p_alpha(alpha: num) -> num:
    x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
    y = np.array([1, 0, 1, alpha, 1, 0, 1], dtype=float)

    a = coefficients(x, y)
    return derivative_at_point_x(a, 7)

def derivative_at_point_x(a: vector , x: num) -> num:
    total = 0.0

    # Schemat Hornera
    # b_k = (k + 1) * a_k+1
    for k in range(len(a) - 1, 0, -1):
        total = total * x + k * a[k]

    return total

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

def coefficients(x: vector, f: vector) -> vector:
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

def main():
    # Funkcja spełnia zależność liniową
    # p_alfa = A*alfa + B

    p0 = p_alpha(0.0) # = B
    p1 = p_alpha(1.0) # = A + B

    A = p1 - p0
    B = p0

    # chcemy żeby funkcja p_alfa, czyli pochodna po Lagrangu się zerowała, więc
    # A * alfa + B = 0
    # alfa = -B / A

    alpha0 = -B / A

    print("Miejsce zerowe p(alfa) = ", alpha0, "")

if __name__ == "__main__":
    main()