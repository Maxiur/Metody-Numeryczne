import numpy as np
import matplotlib.pyplot as plt

num = np.float64

def f(t):
    return np.cos((1 + t) / (t**2 + 0.04))*np.exp(-(t**2))

def romberg(a: num, b: num, tol=1e-8, max_k=25):
    # Tabela Romberga: R[k] przechowuje wiersz przybliżeń dla 2^k przedziałów
    # R[k][0] to wynik złoźonego wzoru trapezów (odpowiada A0,k)
    # R[k][n] to wynik po n-tej ekstrapolacji (odpowiada An,k)

    R = []
    # k = 0: Metoda trapezów dla 1 przedziału
    h = b - a
    R.append([(h / 2) *(f(a) + f(b))])

    for k in range(1, max_k + 1):
        h /= 2

        # z punktów {0, 1} zagęszczamy do {0, 0.5, 1} itd...
        points = np.arange(a + h, b, 2*h)
        trapez_sum = R[k - 1][0] / 2 + h * np.sum(f(points))

        row = [trapez_sum]

        # Ekstrapolacja Richardsona: wypełniamy wiersz po prawej
        for n in range(1, k + 1):
            # Wzór: An,k = (4^n * An-1,k+1 - An-1,k) / (4^n - 1)
            factor = 4**n
            value = (factor * row[n - 1] - R[k - 1][n - 1]) / (factor - 1)
            row.append(value)

        R.append(row)

        # Kryteriu stopu
        if np.abs(R[k][k] - R[k - 1][k - 1]) < tol:
            return R

    return R

def compute_limit(A: num) -> num:
    R = romberg(-A, A, 1e-8)
    return R[-1][-1]

def F(x: num, A: num, limit: num) -> num:
    if x <= -A:
        return 0.0
    if x >= A:
        return limit
    R = romberg(-A, x, 1e-8)
    return R[-1][-1]

def main():
    A = 5.0

    limit = compute_limit(A)
    print(f"Limit funkcji f(x) = cos(x) / (x^2 + 0.04) w przedziale [{-A}, {A}] = {limit}")

    # wykres
    x_i = np.linspace(-A, A, 1000)
    Fx = [F(x, A, limit) for x in x_i]

    plt.plot(x_i, Fx)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Wykres funkcji F(x)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()