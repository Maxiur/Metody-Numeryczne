import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.typing import NDArray

vector = NDArray[np.float64]

@njit(fastmath=True)
def gauss_seidel_step(x: vector, x_old: vector, D: vector):
    n = len(x)

    # 1) i = 0
    i = 0
    x[i] = (1.0 - x_old[i+1] - x_old[i + 4]) / D[i]

    # 2) i = 1, 2, 3
    for i in range(1, 4):
        x[i] = (1.0 - x[i-1] - x_old[i+1] - x_old[i+4]) / D[i]

    # 3) i = 4 .. n-5  (pełna formuła)
    for i in range(4, n - 4):
        x[i] = (1.0 - x[i-1] - x_old[i+1] - x[i-4] - x_old[i+4]) / D[i]

    # 4) i = n-4, n-3, n-2
    for i in range(n - 4, n - 1):
        x[i] = (1.0 - x[i-1] - x[i-4] - x_old[i+1]) / D[i]

    # 5) i = n-1
    i = n - 1
    x[i] = (1.0 - x[i-1] - x[i-4]) / D[i]

def conjugate_gradient(x0:vector, D: vector, iterations: int, tolerance: np.float64) -> vector:
    x = x0.copy()
    r = 1.0 - multiply_A(x, D)
    p = r.copy()
    rTr_old = np.dot(r, r)

    history = [] # lista norm

    for k in range(iterations):
        Ap = multiply_A(p, D)
        alpha = rTr_old / np.dot(p, Ap)

        x_new = x + alpha * p
        r_new = r - alpha * Ap

        history.append(np.linalg.norm(x_new - x))

        if np.linalg.norm(r) < tolerance:
            x = x_new
            break

        rTr_new = np.dot(r_new, r_new)
        beta = rTr_new / rTr_old
        p = r_new + beta * p

        x, r, rTr_old = x_new, r_new, rTr_new
    return x, history

@njit(fastmath=True)
def multiply_A(x: vector, D: vector) -> vector:
    y = D * x
    n = x.shape[0]

    for i in range(n - 4):
        y[i + 1] += x[i]
        y[i] += x[i + 1]

        y[i + 4] += x[i]
        y[i] += x[i + 4]

    for i in range(n - 4 , n - 1):
        y[i + 1] += x[i]
        y[i] += x[i + 1]

    return y

def main() -> None:
    n = 128

    ITERATION_LIMIT = 100
    TOLERANCE = 1e-12

    D: vector = np.full(n, 4.0)
    # --- Niepotrzebne wektory ----
    # L1: vector = np.ones(n - 1, dtype=np.float64)
    # L4: vector = np.ones(n - 4, dtype=np.float64)
    # e: vector = np.ones(n, dtype=np.float64)

    # ---- Gauss-Seidel ----
    x: vector = np.zeros(n, dtype=np.float64)
    x_old: vector = np.zeros(n, dtype=np.float64)
    gauss_history = []

    for iterations in range(1, ITERATION_LIMIT + 1):
        x_old[:] = x
        gauss_seidel_step(x, x_old, D)
        gauss_history.append(np.linalg.norm(x - x_old))

        if np.linalg.norm(x - x_old) < TOLERANCE:
            print(f"Gauss-Seidel Zbieżny po {iterations} iteracjach.")
            break

    # ---- Conjugate Gradient ----
    x0 = np.zeros(n, dtype=np.float64)
    x, cg_history = conjugate_gradient(x0, D, ITERATION_LIMIT, TOLERANCE)
    print(f"CG zakończone po {len(cg_history)} iteracjach.")


    # ---- Rysowanie wykresu ----
    plt.figure()
    plt.plot(gauss_history, label="Gauss-Seidel")
    plt.plot(cg_history, label="Metoda Gradientu sprzężonego")
    plt.yscale("log")
    plt.xlabel("Liczba iteracji")
    plt.ylabel(r"$\|x^{(k)} - x^{(k-1)}\|$")
    plt.title("Porównanie tempa zbieżności: Gauss-Seidel vs. Gradienty sprzężone")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()