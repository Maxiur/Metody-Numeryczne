import numpy as np

num = np.float64

def expected_Lapunow(r: num):
    return np.log(2*r)

def recurvise_function(x: num, r: num):
    return 2*r*x if x < 0.5 else 2*r*(1-x)

def Lapunow(x: num, r: num, N: num = 100000):
    total = 0.0
    for _ in range(N):
        x = recurvise_function(x, r)
        total += np.log(abs(2 * r)) # moduł pochodnej

    return total / N

def main():
    r_values = [3/4, 7/8, 15/16, 31/32]

    for r in r_values:
        l = Lapunow(0.5, r)
        print(f"r={r}, lambda ~ {l:.5f}, lambda_analytical = {expected_Lapunow(r):.5f}")


if __name__ == "__main__":
    main()