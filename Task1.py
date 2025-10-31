import numpy as np

def cholesky_tridiagonal(diagonal: np.ndarray, subdiagonal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = diagonal.shape[0]
    C_diag = np.zeros(n, dtype=np.float64)
    C_subdiag = np.zeros(n - 1, dtype=np.float64)

    C_diag[0] = np.sqrt(diagonal[0])
    for i in range(1, n):
        C_subdiag[i - 1] = subdiagonal[i - 1] / C_diag[i - 1]
        C_diag[i] = np.sqrt(diagonal[i] - C_subdiag[i - 1] ** 2)

    return C_diag, C_subdiag


def forward_substitution_tridiagonal(diagonal: np.ndarray, subdiagonal: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0]
    y = np.zeros(n, dtype=np.float64)

    y[0] = b[0] / diagonal[0]
    for i in range(1, n):
        y[i] = (b[i] - subdiagonal[i - 1] * y[i - 1]) / diagonal[i]

    return y


def backward_substitution_tridiagonal(diagonal: np.ndarray, subdiagonal: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    z = np.zeros(n, dtype=np.float64)

    z[-1] = y[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        z[i] = (y[i] - subdiagonal[i] * z[i + 1]) / diagonal[i]

    return z

def sherman_morrison_formula(z: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    wzór Shermana–Morrisona:
        w = z - (v^T z)/(1 + v^T q) * q
    """

    vTz = np.dot(v, z)
    vTq = np.dot(v, q)
    alpha = float(vTz / (1.0 + vTq))

    return z - alpha * q


def main():
    # A = A1 + uvT
    diagonal = np.array([3, 4, 4, 4, 4, 4, 3], dtype=np.float64)
    subdiagonal = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    b = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
    v = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    u = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)

    # --- Krok (a): A z = b ---
    diag_C, sub_C  = cholesky_tridiagonal(diagonal, subdiagonal)

    y = forward_substitution_tridiagonal(diag_C, sub_C, b)
    z = backward_substitution_tridiagonal(diag_C, sub_C, y)

    # --- Krok (b): A q = u ---
    r = forward_substitution_tridiagonal(diag_C, sub_C, u)
    q = backward_substitution_tridiagonal(diag_C, sub_C, r)

    # --- Krok (c) rozwiązanie ---
    w = sherman_morrison_formula(z, q, v)

    # === ŁADNY PRINT ===
    print("\n" + "=" * 60)
    print(" ROZWIĄZANIE UKŁADU (A1 + u·vᵀ)·w = b ")
    print("=" * 60)
    print(f"{'i':>3} | {'w[i]':>10}")
    print("-" * 18)
    for i, val in enumerate(w, start=1):
        print(f"{i:>3} | {val:>10.10f}")
    print("-" * 18)

    # Macierz wyłącznie do sprawdzenia poprawności rozwiązania
    A1 = np.array([
        [3, 1, 0, 0, 0, 0, 0],
        [1, 4, 1, 0, 0, 0, 0],
        [0, 1, 4, 1, 0, 0, 0],
        [0, 0, 1, 4, 1, 0, 0],
        [0, 0, 0, 1, 4, 1, 0],
        [0, 0, 0, 0, 1, 4, 1],
        [0, 0, 0, 0, 0, 1, 3]
    ], dtype=np.float64)

    A = A1 + np.outer(u, v)
    check = A @ w
    ok = np.allclose(check, b)

    # === Weryfikacja rownania ===
    print("\nWeryfikacja równania A·w ≈ b:\n")
    print(f"{'i':>3} | {'A·w':>10} | {'b':>10} | {'różnica':>10}")
    print("-" * 38)
    for i in range(len(b)):
        diff = check[i] - b[i]
        print(f"{i + 1:>3} | {check[i]:>10.6f} | {b[i]:>10.6f} | {diff:>10.2e}")
    print("-" * 38)
    print(f"\n Układ spełniony: {ok}\n")
    print("=" * 60)

if __name__ == "__main__":
    main()
