import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.complex128]
matrix = NDArray[np.complex128]
num = np.complex128

def horner(coeffs: vector, z: num) -> tuple[num, num, num]:
    # stopień wielomianu
    n = len(coeffs) - 1
    b = coeffs[0]  # P(z)
    c = 0           # P'(z)
    d = 0           # P''(z)

    for i in range(1, n+1):
        d = c + z * d
        c = b + z * c
        b = coeffs[i] + z * b

    return b, c, 2*d

def laguerre(coeffs: vector, z: num, tolerance=1e-9, iterations=100) -> num:
    n = len(coeffs) - 1

    for _ in range(iterations):
        P, P_prime, P_double_prime = horner(coeffs, z)

        # aby nie liczyć dwa razy
        # numerator = n * P
        nP = n * P

        square = (n - 1) * (((n - 1) * P_prime ** 2) - nP * P_double_prime)

        denominator1 = P_prime + square
        denominator2 = P_prime - square
        denominator = denominator1 if abs(denominator1) > abs(denominator2) else denominator2

        z_new = z - nP / denominator

        if abs(z_new - z) < tolerance:
            return z_new

        z = z_new

    return z

def forward_substitution(coeffs: vector, z0: num) -> vector:
    n = len(coeffs) - 1
    b = np.zeros(n, dtype=num)

    b[0] = coeffs[0]
    for i in range(1, n):
        b[i] = coeffs[i] + z0 * b[i-1]

    return b

def main():
    # z^4 + i z^3 - z^2 - i z + 1
    coeffs = np.array([1, 1j, -1, -1j, 1], dtype=num)
    roots = []

    while len(coeffs) > 2:
        z0 = 0 + 0j
        root = laguerre(coeffs, z0)
        roots.append(root)

        coeffs = forward_substitution(coeffs, root)

    quad_roots = np.roots(coeffs)
    roots.extend(quad_roots)

    print("Pierwiastki wielomianu: ")
    for root in roots:
        print(root)

if __name__ == "__main__":
    main()