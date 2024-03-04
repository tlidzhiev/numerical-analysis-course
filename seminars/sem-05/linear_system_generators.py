import numpy as np
from numba import njit


def generate_random_vector(dim: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return np.random.normal(loc=mu, scale=sigma, size=dim)


def generate_random_matrix(dim: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return np.random.normal(loc=mu, scale=sigma, size=(dim, dim))


# SPD matrix generator
def generate_diagonal_matrix(dim: int) -> np.ndarray:
    dmat: np.ndarray = np.zeros((dim, dim))
    dval: np.ndarray = np.exp(generate_random_vector(dim=dim))
    np.fill_diagonal(dmat, dval)
    return dmat


def generate_skew_symmetric_matrix(dim: int) -> np.ndarray:
    smat = generate_random_matrix(dim)
    smat = 0.5 * smat - 0.5 * smat.T
    return smat


@njit(fastmath=True)
def compute_rotation_matrix(smat: np.ndarray, niter: int=100) -> np.ndarray:
    dim, _ = smat.shape
    pmat: np.ndarray = np.eye(dim)
    rmat: np.ndarray = pmat.copy()
    for i in range(niter):
        pmat = np.dot(pmat, smat) / (i + 1)
        rmat = rmat + pmat
    return rmat


def generate_spd_matrix(dim: int) -> np.ndarray:
    dmat = generate_diagonal_matrix(dim=dim)
    smat = generate_skew_symmetric_matrix(dim=dim)
    rmat = compute_rotation_matrix(smat=smat, niter=1000)
    spd_mat = np.dot(rmat.T, np.dot(dmat, rmat))
    spd_mat = 0.5 * spd_mat + 0.5 * spd_mat.T
    return spd_mat


def generate_symmetric_dense_system(dim: int) -> (np.ndarray, np.ndarray, np.ndarray):
    smat = generate_spd_matrix(dim=dim)
    x = generate_random_vector(dim=dim)
    y = np.dot(smat, x)
    return smat, x, y


### System of linear equations with symmetric positivie-definite matrix in sparse format
def generate_random_sparse_matrix(dim: int, frac_off_diag = 0.01) -> (np.ndarray, np.ndarray, np.ndarray):
    n_off_diag = 2 * np.int64(dim * frac_off_diag)
    n_elem = dim + 2 * n_off_diag
    i_idx = np.zeros((n_elem), dtype=np.int64)
    j_idx = np.zeros((n_elem), dtype=np.int64)
    s_mat = np.zeros((n_elem))

    ### generate off-diagonal elements first
    eps = 1.0e-3
    for a in range(dim):
        i_idx[a] = a
        j_idx[a] = a
        s_mat[a] = eps

    ### generate pff-diagonal elements
    idx_counter = dim
    for a in range(n_off_diag):
        idx_0 = np.random.randint(0, dim)
        idx_1 = np.random.randint(0, dim)
        m_val = np.random.normal(0.0, 1.0)
        if idx_0 != idx_1:
            s_mat[idx_0] += 2.0 * np.absolute(m_val)
            s_mat[idx_1] += 2.0 * np.absolute(m_val)

        i_idx[idx_counter] = idx_0
        j_idx[idx_counter] = idx_1
        if idx_0 != idx_1:
            s_mat[idx_counter] += m_val
        idx_counter += 1

        i_idx[idx_counter] = idx_1
        j_idx[idx_counter] = idx_0
        if idx_0 != idx_1:
            s_mat[idx_counter] += m_val
        idx_counter += 1
    return i_idx, j_idx, s_mat


@njit(fastmath=True)
def compute_sparse_matrix_vector_product(
    i_idx: np.ndarray, j_idx: np.ndarray, m_val: np.ndarray, v_val: np.ndarray
) -> np.ndarray:
    v_out = np.zeros(v_val.size)
    n_elem = m_val.size
    for idx_counter in range(n_elem):
        v_out[i_idx[idx_counter]] += m_val[idx_counter] * v_val[j_idx[idx_counter]]
    return v_out


def generate_random_sparse_system(dim: int, frac_off_diag=0.01) -> (np.ndarray, np.ndarray, np.ndarray):
    i_idx, j_idx, s_mat = generate_random_sparse_matrix(dim, frac_off_diag=frac_off_diag)
    x = np.random.normal(0.0, 1.0, dim)
    y = compute_sparse_matrix_vector_product(i_idx=i_idx, j_idx=j_idx, m_val=s_mat, v_val=x)
    return i_idx, j_idx, s_mat, x, y
