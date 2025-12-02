import numpy as np
import scipy
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

def tridiagonalize(A: matrix) -> matrix:
    A = A.copy().astype(np.float64)
    n = A.shape[0]

    for k in range(n - 2):
        x = A[k + 1:, k]

        x_norm = np.linalg.norm(x)
        if x_norm < 1e-14:
            continue

        u = house_holder(x)

        A[k + 1:, k:] -= 2 * np.outer(u, u @ A[k + 1:, k:])
        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:] @ u, u)

    return A

def real_to_complex(v_real):
    n2 = v_real.shape[0]
    n = n2 // 2
    Re = v_real[:n]
    Im = v_real[n:]
    return Re + 1j * Im

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

    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
    T = tridiagonalize(Hf)
    diagonal = np.diagonal(T)
    subdiagonal = np.diagonal(T, offset=1)

    print(T)

    w, v = scipy.linalg.eigh_tridiagonal(diagonal, subdiagonal)
    print(w)
    for i in range(0, 8, 2):
        eigvec = real_to_complex(v[:, i])
        print("v:", eigvec)

if __name__ == "__main__":
    main()