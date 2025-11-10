import numpy as np

def cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    lower = np.zeros_like(matrix, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0

            if i == j:
                # suma kwadratów w wierszu j
                for k in range(j):
                    s += lower[j, k] ** 2
                lower[j, j] = np.sqrt(matrix[j, j] - s)
            else:

                # elementy pod przekątną
                for k in range(j):
                    s += lower[i, k] * lower[j, k]
                lower[i, j] = (matrix[i, j] - s) / lower[j, j]
    return lower

def backward_substitution(matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    z = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            # matrix[i][j]T = matrix[j][i]
            s += matrix[j, i] * z[j]
        z[i] = (y[i] - s) / matrix[i, i]

    return z

def forward_substitution(matrix: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    y = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = 0.0
        for j in range(i):
            s += matrix[i, j] * y[j]
        y[i] = (b[i] - s) / matrix[i, i]

    return y


