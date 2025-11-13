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

def main():
    A: vector = np.array([[19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
                          [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
                          [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
                          [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
                          [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
                          [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12]], dtype=np.float64)
    v, lam1 = power_iteration(A, 1000)
    print(f"lambda: {lam1}")
    print(f"v: {v}")

if __name__ == "__main__":
    main()