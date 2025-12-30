import numpy as np
from numpy.typing import NDArray

num = np.float64
vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def f(x: vector):
    x, y = x
    return 0.25 * x**4 - 0.5 * x**2 + 0.1875 * x + y**2 - 0.0625 * y

def gradient(x: vector) -> vector:
    x, y = x
    return np.array([
        x**3 - x + 0.1875,
        2*y - 0.0625
    ], dtype=num)

def hessian(x: vector) -> matrix:
    x, y = x
    return np.array([
        [3*x**2 - 1, 0],
        [0, 2]
    ], dtype=num)

def levenberg_marquardt_step(x: vector, lam: num, grad: vector, hess: matrix) -> vector:
    hess_local = hess.copy()

    for i in range(len(x)):
        hess_local[i, i] *= (1 + lam)

    delta = np.linalg.solve(hess_local, grad)

    x_new = x - delta

    return x_new

def levenberg_marquardt(x: vector, lam: num, tolerance: float = 1e-10, max_iterations: int = 5000) -> vector:
    for k in range(max_iterations):
        grad = gradient(x)

        # warunek stopu potrzebny
        if np.linalg.norm(grad) < tolerance:
            break

        hess = hessian(x)
        f_x = f(x)

        while True:
            x_test: vector = levenberg_marquardt_step(x, lam, grad, hess)
            f_test = f(x_test)

            if f_test > f_x:
                # krok odrzucony
                lam *= 8
            else:
                # krok zaakceptowany
                x = x_test
                lam /= 8
                break

            if lam > 1e12:
                raise RuntimeError("Nie znaleziono minimum")

    return x, k + 1

def main():
    np.random.seed(0)
    starts = np.random.uniform(-3, 3, size=(128, 2))

    for i, x in enumerate(starts):
        try:
            x_min, iterations = levenberg_marquardt(x, 1/1024)

            print(f"Start {i + 1}: {x}")
            print(f"Koniec: {x_min}")
            print(f"Iteracje: {iterations}")
            print()

        except RuntimeError:
            print(f"Start {i + 1}: {x}")
            print(f"Nie znaleziono minimum")
            print()

if __name__ == "__main__":
    main()