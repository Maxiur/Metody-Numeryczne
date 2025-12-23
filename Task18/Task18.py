import numpy as np
from numpy.typing import NDArray

num = np.float64
vector = NDArray[np.float64]

def horner_value(x: num) -> num:
    # współczynniki z zadania 7
    coeffs = np.array([1.0, -1.0, 0.0, 0.0, 3.0, -2.0, 0.0, 1.0], dtype=num)

    p = coeffs[0]
    for i in range(1, len(coeffs)):
        p = p * x + coeffs[i]
    return p

def f(x: num) -> num:
    return horner_value(x)

def brent(a: num, b: num, c: num, tolerance: float = 1e-6) -> tuple[num, int]:
    prev_interval = abs(c - a)
    prev2_interval = prev_interval

    i = 0
    while True:
        i += 1

        fa, fb, fc = f(a), f(b), f(c)
        old_b = b

        # interpolacja paraboliczna
        denominator = a * (fc - fb) + b * (fa - fc) + c * (fb - fa)

        accept = False
        if abs(denominator) > 1e-14:
            numerator = a ** 2 * (fc - fb) + b ** 2 * (fa - fc) + c ** 2 * (fb - fa)
            d = numerator / (2*denominator)

            if a < d < c:
                new_interval = max(abs(d - a), (c - d))
                if new_interval <= 0.5 * prev2_interval:
                    accept = True

        if not accept:
            d = (c + a) / 2.0

        fd = f(d)

        if fd < fb:
            if d < b:
                c, b = b, d
            else:
                a, b = b, d

        else:
            if d < b:
                a = d
            else:
                c = d

        prev2_interval = prev_interval
        prev_interval = abs(c - a)

        if abs(c - a) < tolerance * (abs(old_b) + abs(d)):
            break

    return b, i

def main():
    a, b, c = -1.0, 0.5, 1.0

    x_min, iterations = brent(a, b, c)
    print(f"Minimum znaleziono w punkcie x = {x_min} w {iterations} iteracjach.")

if __name__ == "__main__":
    main()