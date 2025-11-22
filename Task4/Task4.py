import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def house_holder(x: vector) -> vector:
    # znak x[0]
    sigma = 1.0 if x[0] >= 0 else -1.0

    # odbicie wektora
    v = x.copy()
    x_norm = np.linalg.norm(x)
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

        # P=I−2uuT
        # PA=A−2u(uTA)
        # AP=A−2(Au)uT

        A[k + 1:, k:] -= 2 * np.outer(u, u @ A[k + 1:, k:])
        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:] @ u, u)

    return A

def qr_step_tridiagonal(T: matrix) -> matrix:
    n = T.shape[0]
    T = T.copy()

    # QR T = Q R za pomocą rotacji Givensa (zerujemy elementy poddiagonalne)
    for i in range(n - 1):
        a = T[i, i]
        b = T[i + 1, i]

        r = np.hypot(a, b)
        c = a / r
        s = b / r

        apply_givens_rows(T, i, i+1, c, s)
        apply_givens_cols(T, i, i+1, c, s)

    return T

def apply_givens_rows(T, i, j, c, s):
    """
    Zastosowanie rotacji Givensa G na wierszach i,j: T := G @ T.
    """
    n = T.shape[0]

    left = max(0, i - 1)
    right = min(n - 1, i + 2)

    for col in range(left, right + 1):
        Ti = T[i, col]
        Tj = T[j, col]

        T[i, col] = c * Ti + s * Tj
        T[j, col] = -s * Ti + c * Tj

def apply_givens_cols(T, i, j, c, s):

    """
    Zastosowanie rotacji Givensa G na kolumnach i,j: T := T @ G^T.
    """
    n = T.shape[0]

    top = max(0, i - 1)
    bottom = min(n - 1, i + 2)

    for row in range(top, bottom + 1):
        Ti = T[row, i]
        Tj = T[row, j]

        T[row, i] = c * Ti + s * Tj
        T[row, j] = -s * Ti + c * Tj


def main():
    A: matrix = np.array([[19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
                          [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
                          [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
                          [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
                          [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
                          [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12]], dtype=np.float64)

    T = tridiagonalize(A)
    np.set_printoptions(linewidth=np.inf)
    print(T)
    print("\n\nAfter diagonal ------------------------------\n\n")
    eps = 1e-12
    max_iter = 1000

    for it in range(max_iter):
        diag_old = np.diag(T).copy()
        T = qr_step_tridiagonal(T)
        diag_new = np.diag(T)

        # sprawdzamy maksymalną zmianę przekątnej
        diff = np.linalg.norm(diag_new - diag_old)
        if diff < eps:
            print(f"Diagonal converged after {it} iterations, diff = {diff:.2e}")
            break
    print(T)

if __name__ == "__main__":
    main()