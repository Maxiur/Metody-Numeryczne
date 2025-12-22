import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def g(x: vector) -> vector:
    X = x[0]
    Y = x[1]

    return np.array([
        2*X**2 + Y**2 - 2, # f1(x)
        (X - 0.5)**2 + (Y - 1)**2 - 0.25], # f2(x)
        dtype=num)

def jacobian(x: vector) -> matrix:
    # Jacobian policzony analitycznie!
    X = x[0]
    Y = x[1]
    return np.array([
        [4*X, 2*Y],
        [2*(X-0.5), 2*(Y-1)]
    ], dtype=num)

def newton_iteration(x: vector) -> vector:
    J = jacobian(x)
    F = g(x)

    # J * delta = -F
    delta = np.linalg.solve(J, F)
    return x - delta

def newton(x: vector, tolerance: float = 1e-10, iterations: int = 50) -> tuple[vector, int]:

    for i in range(iterations):
        x_next = newton_iteration(x)

        if np.linalg.norm(x_next - x) < tolerance:
            return x_next, i + 1

        x = x_next

    return x, iterations


def main():
    # punkty startowe obejmujące wszystkie możliwe pierwiastki
    guesses = [
        np.array([0, 0.5]),
        np.array([0, 1.5]),
        np.array([1, 0.5]),
        np.array([1, 1.5])
    ]

    roots = []

    for x0 in guesses:
        root, iterations = newton(x0)
        # sprawdzenie unikalności
        if not any(np.allclose(root, r, atol=1e-8) for r in roots):
            roots.append(root)
            print(f"Znaleziono pierwiastek x = {root} w {iterations} iteracjach")

if __name__ == "__main__":
    main()