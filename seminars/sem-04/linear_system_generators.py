import numpy as np


def generate_diagonally_dominant_matrix(dim: int) -> np.ndarray:
    eps = 1.0e-3
    smat = np.random.normal(loc=0.0, scale=1.0, size=(dim, dim))
    np.fill_diagonal(smat, np.absolute(smat).sum(axis=1) + eps)
    return smat


def generate_diagonally_dominant_linear_system_equations(dim: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_diagonally_dominant_matrix(dim=dim)
    x = np.random.normal(loc=0.0, scale=1.0, size=dim)
    y = np.dot(smat, x)
    return smat, x, y


def generate_matrix_tridiag(dim: int) -> np.ndarray:
    eps = 1.0e-6
    smat = (2.0 + eps) * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)
    smat[0, -1] = smat[-1, 0] = -1.0
    return dim * dim * smat


def generate_tridiag_linear_system_equations(dim: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_matrix_tridiag(dim=dim)
    x = np.random.normal(loc=0.0, scale=1.0, size=dim)
    y = np.dot(smat, x)
    return smat, x, y


def generate_weak_diagonally_dominant_matrix(dim: int) -> np.ndarray:
    eps = 1.0e-6
    smat = np.random.normal(loc=0.0, scale=1.0, size=(dim, dim))
    wmat = np.zeros((dim, dim))
    kappa = 0.8
    for idx0 in range(dim):
        for idx1 in range(dim):
            wmat[idx0, idx1] = np.exp(-kappa * np.absolute((idx1 - idx0)))
    smat = wmat * smat
    np.fill_diagonal(smat, 0.5 * np.sum(np.absolute(smat), axis=1) + eps)
    return smat


def generate_weak_diagonally_dominant_linear_system_equations(dim: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_weak_diagonally_dominant_matrix(dim=dim)
    x = np.random.normal(loc=0.0, scale=1.0, size=dim)
    y = np.dot(smat, x)
    return smat, x, y
