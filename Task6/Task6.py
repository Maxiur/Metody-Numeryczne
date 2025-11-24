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

def thomas_solve(L: vector, U: vector, b: vector) -> vector:
    n = len(U)

    # forward
    for i in range(1, n):
        b[i] -= L[i - 1] * b[i - 1]

    # back
    x = np.zeros(n)
    x[-1] = b[-1] / U[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - L[i] * x[i + 1]) / U[i]

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

def inverse_iteration_with_sherman(L: vector, U: vector, u: vector, iterations: int = 1000) -> vector:
    n = len(U)
    y = np.random.rand(n)
    y /= np.linalg.norm(y)

    for k in range(iterations):
        T_inverse_y = thomas_solve(L.copy(), U.copy(), y.copy())
        T_inverse_u = thomas_solve(L.copy(), U.copy(), u.copy())

        z = sherman_morrison_formula(T_inverse_y, T_inverse_u, u)

        y = z / np.linalg.norm(z)

    return y

def brute_force_eigenvector(A: matrix, tau: float) -> tuple[vector, float, int]:
    """
    Znajduje wektor własny odpowiadający wartości własnej najbliższej tau
    poprzez pełne diagonalizację macierzy (brute force).

    Args:
        A: Macierz symetryczna
        tau: Przybliżona wartość własna

    Returns:
        eigenvector: Wektor własny dla najbliższej wartości własnej
        eigenvalue: Najbliższa wartość własna
        index: Indeks znalezionej wartości własnej
    """
    # Oblicz wszystkie wartości i wektory własne
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Znajdź indeks najbliższej wartości własnej do tau
    index = np.argmin(np.abs(eigenvalues - tau))

    # Wyodrębnij odpowiadający wektor własny
    eigenvector = eigenvectors[:, index]
    eigenvalue = eigenvalues[index]

    # Upewnij się, że pierwszy niezerowy element jest dodatni (dla jednoznaczności znaku)
    for i in range(len(eigenvector)):
        if abs(eigenvector[i]) > 1e-10:
            if eigenvector[i] < 0:
                eigenvector = -eigenvector
            break

    return eigenvector, eigenvalue, index

def main():
    v: vector = np.array([1, 0, 0, 0, 1], dtype=np.float64)
    u: vector = np.array([1, 0, 0, 0, 1], dtype=np.float64)

    A = np.array([
        [2, -1, 0, 0, 1],
        [-1, 2, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 2, -1],
        [1, 0, 0, -1, 2],
    ], dtype=np.float64)

    # M = T + u*vT
    diagonal = np.array([1, 2, 1, 2, 1], dtype=np.float64)
    subdiagonal = np.array([-1, 1, 1, -1], dtype=np.float64)

    tau = 0.38197
    diagonal_T = diagonal - tau

    L, U = thomas_factor(subdiagonal, diagonal_T)

    np.set_printoptions(linewidth=np.inf)
    eigenvector = inverse_iteration_with_sherman(L, U, u)
    print("Przybliżony wektor własny dla λ ≈ 0.38197:")
    print(eigenvector)

    eigenvector, eigenvalue, index = brute_force_eigenvector(A, tau)
    print(eigenvalue)
    print(eigenvector)


if __name__ == '__main__':
    main()