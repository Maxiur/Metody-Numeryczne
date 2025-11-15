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

def qr_step_tridiagonal(T: matrix) -> matrix:
    n = T.shape[0]
    T = T.copy()

    return T

def main():
    A: matrix = np.array([[19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
                          [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
                          [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
                          [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
                          [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
                          [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12]], dtype=np.float64)

    T = tridiagonalize(A)
    np.set_printoptions(linewidth=np.inf)
    for _ in range(5000):
        T = qr_step_tridiagonal(T)
    print(T)


if __name__ == "__main__":
    main()