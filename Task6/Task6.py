import numpy as np
from numpy.typing import NDArray

vector = NDArray[np.float64]
matrix = NDArray[np.float64]

def main():
    v: vector = np.array([1, 0, 0, 0, 1], dtype=np.float64)
    u: vector = np.array([1, 0, 0, 0, 1], dtype=np.float64)


    A = np.array([
        [2, -1, 0, 0, 1],
        [-1, 2, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 2, -1],
        [1, 0, 0, -1, 2],
    ], dtype=np.float64)

    diagonal = np.array([2, 2, 1, 2, 2], dtype=np.float64)
    subdiagonal = np.array([-1, 1, 1, -1], dtype=np.float64)

    tau = 0.38197
    T = diagonal - tau


if __name__ == '__main__':
    main()