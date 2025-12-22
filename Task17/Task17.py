import numpy as np

num = np.float64

def f(x: num) -> num:
    return 0.25 * x**4 - 0.5 * x**2 - (1/16) * x

def golden_ratio(a: num, b: num, c: num, tolerance: float = 1e-6) -> tuple[num, int]:
    w = (3 - np.sqrt(5)) / 2

    i = 0
    while True:
        i += 1
        if abs(b - a) > abs(c - b):
            d = a + w * abs(b - a)
        else:
            d = b + w * abs(c - b)

        if abs(c - a) < tolerance * (abs(b) + abs(d)):
            break

        fb, fd = f(b), f(d)

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

    return b, i

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
    a, b, c = 0.0, 1.0, 3.0
    tolerance = 1e-6

    gold_min, gold_iterations = golden_ratio(a, b, c, tolerance)
    brent_min, brent_iterations = brent(a, b, c, tolerance)

    print(f"Złoty podział: x = {gold_min:.8f}, iteracje = {gold_iterations}")
    print(f"Metoda Brenta: x = {brent_min:.8f}, iteracje = {brent_iterations}")

if __name__ == "__main__":
    main()
