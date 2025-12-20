import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]
num = np.float64

def f(x: num) -> num:
    return np.sin(np.pi * (1 + np.sqrt(x)) / (1 + x**2)) * np.exp(-x)

def romberg(a: num, b: num, tol=1e-7, max_k=25):
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
        n_new_points = 2**(k - 1)
        points = a + h * (2*np.arange(1, n_new_points + 1) - 1)
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

def main():
    # exp(-A) < 10^-7
    # Wyznaczone A ~ -16.118
    A = 17.0

    tabelau = romberg(0, A)

    for i, row in enumerate(tabelau):
        print(f"k={i}: " + " ".join(f"{val:.10f}" for val in row))

    print(tabelau[-1][-1])

if __name__ == "__main__":
    main()