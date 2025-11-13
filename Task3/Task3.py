import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def power_iteration(A: matrix, iterations: int) -> tuple[vector, np.float64]:
    y = np.random.rand(A.shape[0])
    y = y / np.linalg.norm(y)

    for _ in range(iterations):
        # obliczenie wektora Ay
        z = A @ y

        # obliczenia normy
        z_norm = np.linalg.norm(z)

        # normalizacja wektora
        y = z / z_norm

    lam = float(np.linalg.norm(z))

    return y, lam

def second_eigenpar(A: matrix, e1: vector, iterations: int) -> tuple[vector, np.float64]:

    y = np.random.rand(A.shape[0])
    y -= e1 * (e1 @ y)  # rzutowanie
    y = y / np.linalg.norm(y)

    for i in range(iterations):
        z = A @ y

        # ortogonalizacja względem e1
        z = z - e1 * (e1 @ z)

        y = z / np.linalg.norm(z)

    lam = float(np.linalg.norm(z))

    return y, lam


def main():
    A: vector = np.array([[19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
                          [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
                          [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
                          [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
                          [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
                          [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12]], dtype=np.float64)
    e1, lam1 = power_iteration(A, 1000)
    e2, lam2 = second_eigenpar(A, e1, 1000)
    print(f"-------------------------------")
    print(f"lambda: {lam1}")
    print(f"e1: {e1}")
    print(f"lambda: {lam2}")
    print(f"e2: {e2}")

if __name__ == "__main__":
    main()