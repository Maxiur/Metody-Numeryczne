import scipy
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

def tridiagonalize(A: matrix):
    A = A.copy().astype(np.float64)
    n = A.shape[0]
    Q = np.eye(n, dtype=np.float64)


    householder_vectors = []
    for k in range(n - 2):
        x = A[k + 1:, k]
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-14:
            continue
        u = house_holder(x)

        u_full = np.zeros(n)
        u_full[k + 1:] = u
        householder_vectors.append(u_full)

        A[k + 1:, k:] -= 2 * np.outer(u, u @ A[k + 1:, k:])
        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:] @ u, u)

    return A, householder_vectors

def real_to_complex(w: vector) -> NDArray[np.complex128]:
    n2 = len(w)
    n = n2 // 2
    x = w[:n]
    y = w[n:]
    u = x + 1j * y
    u_norm = np.sqrt(np.vdot(u, u))
    if u_norm == 0.0:
        return u
    return u / u_norm

def apply_householders(y: matrix, householder_vectors: list[vector]) -> matrix:
    # Wektory własne Hf: w = P_{n-2} * ... * P_1 * y
    w = y.copy()

    # Stosujemy wektory Householdera w ODWRÓCONEJ KOLEJNOŚCI
    # do macierzy wektorów własnych y.
    for u_full in reversed(householder_vectors):
        w -= 2 * np.outer(u_full, u_full @ w)

    return w

def main():
    H = np.array([
        [0,   1,   0, -1j],
        [1,   0, -1j,   0],
        [0,  1j,   0,   1],
        [1j,  0,   1,   0]
    ], dtype=np.complex128)

    A = np.real(H)
    B = np.imag(H)

    Hf: matrix = np.block([
        [A, -B],
        [B, A]
    ]).astype(np.float64)

    T, householder_vectors = tridiagonalize(Hf)

    np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)

    diagonal = np.diag(T)

    subdiagonal_raw = np.diag(T, k=-1)

    subdiagonal = np.where(np.abs(subdiagonal_raw) < 1e-12, 0.0, subdiagonal_raw)

    v, y = scipy.linalg.eigh_tridiagonal(diagonal, subdiagonal)

    # Przekształcenie wektorów własnych Hf: w = Q @ y
    w = apply_householders(y, householder_vectors)

    print("## Wartości Własne Hf (rzeczywiste):")
    print(v)

    print("---")
    print("## Wektory Własne H (kolumny):")

    # Konwersja wektorów własnych Hf do H
    for i in [0, 2, 4, 6]:
        eigvec_h = real_to_complex(w[:, i])
        print(f"  λ={v[i]:.4f}: {eigvec_h}")

if __name__ == "__main__":
    main()
