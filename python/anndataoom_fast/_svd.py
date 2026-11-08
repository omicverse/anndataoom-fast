"""Randomized SVD over the implicit scaled operator.

Same algorithm as ``sklearn.utils.extmath.randomized_svd`` (Halko et al.
2011, Algorithm 4.4) with the same Gaussian start drawn from
``RandomState(random_state)``, but run entirely in float64 and made
thread-count independent:

* ``A @ Q`` and ``A.T @ Q`` are the C++ operator products (fixed-order
  reductions);
* each power iteration is normalised with a deterministic CholeskyQR2 instead
  of LAPACK's LU / Householder QR (multithreaded LAPACK does not promise
  bit-identical results across thread counts).  Normalisation only changes the
  basis, not the subspace, so the singular triplets agree with sklearn's up to
  rounding;
* the final small SVD runs on single-threaded LAPACK.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from . import _core


def _c64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def randomized_svd(op, n_components: int, *, n_oversamples: int = 0, n_iter: int = 4,
                   random_state=0):
    from threadpoolctl import threadpool_limits

    if isinstance(random_state, np.random.RandomState):
        rs = random_state
    else:
        rs = np.random.RandomState(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = op.shape
    transpose = n_samples < n_features
    if transpose:
        mat, rmat, width = op.rmatmul, op.matmul, n_samples
    else:
        mat, rmat, width = op.matmul, op.rmatmul, n_features

    Q = _c64(rs.normal(size=(width, n_random)))
    for _ in range(n_iter):
        Q = _core.orthonormalize(mat(Q))
        Q = _core.orthonormalize(rmat(Q))
    Q = _core.orthonormalize(mat(Q))
    B = rmat(Q).T  # Q.T @ M
    with threadpool_limits(1):
        Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver="gesdd")
    U = _core.gemm_tall(Q, _c64(Uhat))
    if transpose:
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    return U[:, :n_components], s[:n_components], Vt[:n_components, :]
