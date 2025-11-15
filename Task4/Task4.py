import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def house_holder(x: vector) -> vector:
    x_norm = np.linalg.norm(x)

    # znak x[0]
    sigma = 1.0 if x[0] >= 0 else -1.0

    # odbicie wektora
    v = x.copy()
    v[0] += sigma * x_norm

    # normalizacja wektora Householdera u
    v_norm = np.linalg.norm(v)
    u = v / v_norm

    return u

def tridiagonalize(A: matrix) -> tuple[matrix, matrix]:
    A = A.copy()
    n = A.shape[0]

    for k in range(n - 2):
        # opuszczamy pierwszy element
        x = A[k + 1:, k]

        u = house_holder(x)

        # tworzenie macierzy Householdera
        H_small = np.eye(n - k - 1) - 2.0 * np.outer(u, u)

        # wstawiamy H_small w odpowiednie miejsce dużej macierzy Q_k
        Q_k = np.eye(n)
        Q_k[k + 1:, k + 1:] = H_small

        # transformacja podobieństwa
        A = (Q_k.T @ A) @ Q_k

    return A

def givens(a, b):
    if b == 0:
        return 1.0, 0.0

    r = np.hypot(a, b)
    c = a / r
    s = -b / r

    return c, s

def qr_tridiagonal_givens(T: matrix) -> matrix:
    R = T.copy()
    n = R.shape[0]

    for i in range(n - 1):
        a = R[i, i]
        b = R[i + 1, i]
        d = R[i + 1, i + 1]
        e = R[i + 2, i + 1] if i + 2 < n else 0.0

        # jeżeli już wyzerowany to pomiń
        if b == 0.0:
            continue

        r = np.hypot(a, b)

        # aktualizujemy tylko elementy trójdiagonalne
        R[i, i] = r
        R[i + 1, i] = 0.0

        R[i, i + 1] = b * (a + d) / r
        R[i + 1, i + 1] = (a * d - b * b) / r

        # elementy powstające z przesunięcia trójdiagonali
        if i + 2 < n:
            R[i, i + 2] = b * e / r
            R[i + 1, i + 2] = a * e / r

    return R


def main():
    A: matrix = np.array([[19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
                          [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
                          [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
                          [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
                          [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
                          [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12]], dtype=np.float64)

    T = tridiagonalize(A)
    np.set_printoptions(linewidth=np.inf)
    T = qr_tridiagonal_givens(T)
    print(T)


if __name__ == "__main__":
    main()