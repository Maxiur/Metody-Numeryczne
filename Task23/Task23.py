import numpy as np
from macholib.mach_o import dylib
from numpy.typing import NDArray

num = np.float64
vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def rosenbrock(x: vector) -> num:
    f = (1 - x[0])**2

    for i in range(len(x) - 1):
        f += 100*(x[i + 1] - x[i]**2)**2

    return f

def gradient(x: vector) -> vector:
    n = len(x)
    grad = np.zeros_like(x, dtype=num)

    grad[0] = -2*(1 - x[0]) - 400*(x[1] - x[0]**2) * x[0]

    for i in range(1, n - 1):
        grad[i] = (
            200*(x[i] - x[i - 1]**2)
            - 400*(x[i + 1] - x[i]**2) * x[i]
            - 2*(1 - x[i])
        )

    grad[-1] = 200*(x[-1] - x[-2]**2)
    return grad

def hessian(x: vector) -> matrix:
    n = len(x)
    H = np.zeros((n, n), dtype=num)

    for i in range(n - 1):
        H[i, i] += 2 + 800 * x[i]**2 - 400 * (x[i + 1] - x[i]**2)
        H[i, i + 1] = -400 * x[i]
        H[i + 1, i] = H[i, i + 1] # -400 * x[i]

    for i in range(n):
        H[i, i] += 200

    return H

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
        f = rosenbrock(x)

        while True:
            x_test: vector = levenberg_marquardt_step(x, lam, grad, hess)
            f_test = rosenbrock(x_test)

            if f_test > f:
                # krok odrzucony
                lam *= 8
            else:
                # krok zaakceptowany
                x = x_test
                lam /= 8
                break

            if lam > 1e12:
                break

    return x

def main():
    np.random.seed(0)

    for i in range(6):
        x = np.random.uniform(-2, 2, size=4)

        x_min: vector = levenberg_marquardt(x, 1/1024)

        print(f"Start {i + 1}: {x}")
        print(f"Koniec: {x_min}")
        print()


if __name__ == "__main__":
    main()
