import numpy as np


def generate_diagonally_dominant_matrix(n: int) -> np.ndarray:
    eps = 1.0e-3
    smat = np.random.normal(loc=0.0, scale=1.0, size=(n, n))
    for i in range(n):
        smat[i, i] = np.sum(np.absolute(smat[i, :])) + eps
    return smat


def generate_diagonally_dominant_linear_system_equations(n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_diagonally_dominant_matrix(n=n)
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = np.dot(smat, x)
    return smat, x, y


def generate_matrix_tridiag(n: int) -> np.ndarray:
    eps = 1.0e-6
    smat = np.zeros((n, n))
    for i in range(n):
        smat[i, i] = 2.0 + eps
        if i > 0:
            smat[i, i - 1] = -1.0
            smat[i - 1, i] = -1.0
    smat[0, -1] = -1.0
    smat[-1, 0] = -1.0
    return n * n * smat


def generate_tridiag_linear_system_equations(n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_matrix_tridiag(n=n)
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = np.dot(smat, x)
    return smat, x, y


def generate_weak_diagonally_dominant_matrix(n: int) -> np.ndarray:
    eps = 1.0e-6
    smat = np.random.normal(loc=0.0, scale=1.0, size=(n, n))
    wmat = np.zeros((n, n))
    kappa = 0.8
    for idx0 in range(n):
        for idx1 in range(n):
            wmat[idx0, idx1] = np.exp(-kappa * np.absolute((idx1 - idx0)))
    smat = wmat * smat
    for a in range(n):
        smat[a, a] = 0.5 * np.sum(np.absolute(smat[a, :])) + eps
    return smat


def generate_weak_diagonally_dominant_linear_system_equations(n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_weak_diagonally_dominant_matrix(n=n)
    x = np.random.normal(loc=0.0, scale=1.0, size=n)
    y = np.dot(smat, x)
    return smat, x, y
