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

        if np.allclose(x[1:], 0.0):
            continue

        u = house_holder(x)

        A[k + 1:, k:] -= 2.0 * np.outer(u, u @ A[k + 1:, k:])
        A[:, k + 1:] -= 2.0 * np.outer(A[:, k + 1:] @ u, u)

        Q[:, k + 1:] -= 2.0 * np.outer(Q[:, k + 1:] @ u, u)


    A[np.abs(A) < 1e-14] = 0.0
    Q[np.abs(Q) < 1e-14] = 0.0
    return A, Q

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

    np.set_printoptions(linewidth=np.inf)
    print(T)
    print(Q)


if __name__ == "__main__":
    main()