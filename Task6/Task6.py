import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

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


def sherman_morrison_formula(z: vector, q: vector, v: vector) -> vector:
    """
    wzór Shermana–Morrisona:
        w = z - (v^T z)/(1 + v^T q) * q
    """

    vTz = np.dot(v, z)
    vTq = np.dot(v, q)
    alpha = float(vTz / (1.0 + vTq))

    return z - alpha * q

def inverse_iteration_with_sherman(subdiagonal: vector, diagonal: vector, u: vector, iterations: int = 1000) -> vector:
    n = len(diagonal)
    L, U = thomas_factor(subdiagonal, diagonal)
    superdiagonal = subdiagonal

    y = np.random.rand(n)
    y /= np.linalg.norm(y)

    T_inverse_u = thomas_solve(L, U, u, superdiagonal)
    for k in range(iterations):
        T_inverse_y = thomas_solve(L, U, y, superdiagonal)

        z = sherman_morrison_formula(T_inverse_y, T_inverse_u, u)

        y = z / np.linalg.norm(z)

    return y

def main():
    # u = v
    u: vector = np.array([1, 0, 0, 0, 1], dtype=np.float64)

    # M = T + u*vT
    diagonal = np.array([1, 2, 1, 2, 1], dtype=np.float64)
    subdiagonal = np.array([-1, 1, 1, -1], dtype=np.float64)

    tau = 0.38197
    diagonal_T = diagonal - tau

    np.set_printoptions(linewidth=np.inf)
    eigenvector = inverse_iteration_with_sherman(subdiagonal, diagonal_T, u)
    print("Przybliżony wektor własny dla λ ≈ 0.38197:")
    print(eigenvector)

if __name__ == '__main__':
    main()