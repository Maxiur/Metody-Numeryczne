import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def house_holder(x: vector) -> vector:
    sigma = 1.0 if x[0] >= 0 else -1.0

    v = x.copy()
    x_norm = np.linalg.norm(x)
    if x_norm == 0.0:
        return np.zeros_like(x)

    v[0] += sigma * x_norm

    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return np.zeros_like(x)

    u = v / v_norm

    return u

def tridiagonalize(A: matrix) -> tuple[matrix, matrix]:
    A = A.copy().astype(np.float64)
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)

    for k in range(n - 2):
        x = A[k + 1:, k]

        x_norm = np.linalg.norm(x)
        if x_norm < 1e-14:
            continue

        u = house_holder(x)

        A[k + 1:, k:] -= 2 * np.outer(u, u @ A[k + 1:, k:])

        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:] @ u, u)

        Q[:, k + 1:] -= 2 * np.outer(Q[:, k + 1:] @ u, u)


    A[np.abs(A) < 1e-14] = 0.0
    Q[np.abs(Q) < 1e-14] = 0.0
    return A, Q


def qr_step_tridiagonal(T: matrix, P: matrix) -> matrix:
    n = T.shape[0]
    T = T.copy()

    for i in range(n - 1):
        a = T[i, i]
        b = T[i + 1, i]

        if abs(b) < 1e-15:
            continue

        r = np.hypot(a, b)
        c = a / r
        s = b / r

        # 1. T = G T
        apply_givens_rows(T, i, i+1, c, s)

        # 2. T = T G^T
        apply_givens_cols(T, i, i+1, c, s)

        # 3. P = P G
        apply_givens_cols(P, i, i+1, c, -s)

    return T, P


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


def qr_algorithm(T: matrix, Q: matrix, max_iter: int = 500, eps: float = 1e-12):
    P = Q.copy()
    n = T.shape[0]

    for it in range(1, max_iter + 1):
        T, P = qr_step_tridiagonal(T, P)

        # norma elementów pod diagonalą
        off = np.linalg.norm(np.diagonal(T, -1))

        if off < eps:
            return T, P, it

    # jeśli nie zbiegło — zwracamy info
    return T, P, max_iter

def main():
    H = np.array([
        [0,   1,   0,  -1j],
        [1,   0,  -1j,  0],
        [0,   1j,  0,   1],
        [1j,  0,   1,   0]
    ], dtype=np.complex128)

    A = np.real(H)
    B = np.imag(H)

    Hf: matrix = np.block([
        [A, -B],
        [B, A]
    ]).astype(np.float64)

    T, Q = tridiagonalize(Hf)

    Diagonal, P, steps = qr_algorithm(T, Q)

    np.set_printoptions(linewidth=np.inf)
    print(np.linalg.eigvals(T))
    print(np.linalg.eigvals(Hf))

    print(Diagonal)
    print()
    print(P)

if __name__ == "__main__":
    main()