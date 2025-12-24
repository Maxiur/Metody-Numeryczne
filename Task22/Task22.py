import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

num = np.float64
vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def rosenbrock(x: num, y: num) -> num:
    return (1 - x)**2 + 100*(y - x**2)**2

def gradient(x: num, y: num) -> vector:
    df_dx = -2*(1 - x) - 400*(y - x**2)*x
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy], dtype=num)

def hessian(x: num, y: num) -> matrix:
    h11 = 2 - 400*x*(y - x**2) + 800*x**2
    h12 = -400*x
    h22 = 200
    return np.array([[h11, h12], [h12, h22]], dtype=num)

def levenberg_marquardt_step(x: num, y: num, lam: num, grad: vector, hess: matrix) -> tuple[num, num]:
    hess_local = hess.copy()

    for i in range(2):
        hess_local[i, i] *= (1 + lam)

    delta = np.linalg.solve(hess_local, grad)

    x_new = x - delta[0]
    y_new = y - delta[1]

    return x_new, y_new

def levenberg_marquardt(x: num, y: num, lam: num, tolerance: float = 1e-10, max_iterations: int = 5000) -> tuple[num, num, vector]:
    path = [(x, y)]

    for k in range(max_iterations):
        grad = gradient(x, y)

        # warunek stopu potrzebny
        if np.linalg.norm(grad) < tolerance:
            break

        hess = hessian(x, y)
        f = rosenbrock(x, y)

        while True:
            x_test, y_test = levenberg_marquardt_step(x, y, lam, grad, hess)
            f_test = rosenbrock(x_test, y_test)

            if f_test > f:
                # krok odrzucony
                lam *= 8
            else:
                # krok zaakceptowany
                x = x_test
                y = y_test
                path.append((x, y))
                lam /= 8
                break

            if lam > 1e12:
                break

    return x, y, np.array(path)

def plot_all_paths(all_paths: list[vector]):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, levels=100, cmap="grey")

    # Rysujemy wszystkie trajektorie
    for path in all_paths:
        plt.plot(path[:, 0], path[:, 1], "-o", markersize=3)

    # Minimum
    plt.plot(1, 1, "b*", markersize=12)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajektorie LM z różnych startów")
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(0)
    all_paths = []

    for i in range(5):
        x, y = np.random.uniform(-2, 2, size=2)

        x_min, y_min, path = levenberg_marquardt(x, y, 1/1024)
        all_paths.append(path)

        print(f"Start {i + 1}: ({x:.3f}, {y:.3f})")
        print(f"  Koniec: ({x_min:.6f}, {y_min:.6f})")
        print(f"  Kroki: {len(path)}")
        print()

    plot_all_paths(all_paths)

if __name__ == "__main__":
    main()