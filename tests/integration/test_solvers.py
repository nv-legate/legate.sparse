# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cunumeric as np
import numpy
import pytest
import scipy
import scipy.sparse as scpy
import scipy.stats as stats

import sparse.linalg as linalg
from sparse import csr_array, eye


class Normal(stats.rv_continuous):
    def _rvs(self, *args, size=None, random_state=None):
        return random_state.standard_normal(size)


def sample(N: int, D: int, density: float, seed: int):
    NormalType = Normal(seed=seed)
    SeededNormal = NormalType()
    return scpy.random(
        N,
        D,
        density=density,
        format="csr",
        dtype=np.float64,
        random_state=seed,
        data_rvs=SeededNormal.rvs,
    )


def sample_dense_vector(N: int, density: float, seed: int):
    return numpy.array(sample(N, 1, density, seed).todense()).squeeze()


def test_cg_solve():
    N, D = 1000, 1000
    seed = 471014
    A = sample(N, D, 0.1, seed).todense()
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred, iters = linalg.cg(A, y, tol=1e-8)
    assert np.allclose((A @ x_pred), y)


def test_cg_solve_with_callback():
    N, D = 100, 100
    seed = 471014
    A = sample(N, D, 0.1, seed).todense()
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    residuals = []

    def callback(x):
        # Test that nothing goes wrong if we do some arbitrary computation in
        # the callback on x.
        residuals.append(y - A @ x)

    x_pred, iters = linalg.cg(A, y, tol=1e-8, callback=callback)
    assert np.allclose((A @ x_pred), y)
    assert len(residuals) > 0


def test_cg_solve_with_identity_preconditioner():
    N, D = 1000, 1000
    seed = 471014
    A = sample(N, D, 0.1, seed).todense()
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred, iters = linalg.cg(A, y, M=eye(A.shape[0]), tol=1e-8)
    assert np.allclose((A @ x_pred), y)


def test_cg_solve_with_linear_operator():
    N, D = 100, 100
    seed = 471014
    A = sample(N, D, 0.1, seed).todense()
    A = 0.5 * (A + A.T)
    A = A + N * np.eye(N)
    # Assert that A is indeed positive semi-definite.
    assert np.all(np.linalg.eigvals(A) > 0)
    A = csr_array(A)
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)

    def matvec(x):
        return A @ x

    x_pred, iters = linalg.cg(
        linalg.LinearOperator(A.shape, matvec=matvec), y, tol=1e-8
    )
    assert np.allclose((A @ x_pred), y)

    def matvec(x, out=None):
        return A.dot(x, out=out)

    x_pred, iters = linalg.cg(
        linalg.LinearOperator(A.shape, matvec=matvec), y, tol=1e-8
    )
    assert np.allclose((A @ x_pred), y)


def test_bicg_solve():
    N, D = 100, 100
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred = linalg.bicg(A, y, tol=1e-8)
    assert np.allclose((A @ x_pred), y)


@pytest.mark.skip(reason="BiCGSTAB still doesn't work")
def test_bicgstab_solve():
    N, D = 100, 100
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A.dot(x)), y)
    x_pred = linalg.bicgstab(A, y, tol=1e-8)
    assert np.allclose(A.dot(x_pred), y)


def test_cgs_solve():
    N, D = 100, 100
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred = linalg.cgs(A, y, tol=1e-8)
    assert np.allclose((A @ x_pred), y, rtol=1e-5, atol=1e-6)


def test_lsqr_solve():
    N, D = 1000, 500
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred = linalg.lsqr(A, y, atol=1e-10, btol=1e-10)[0]
    assert np.allclose((A @ x_pred), y)


@pytest.mark.xfail(reason="Seems to be failing on 2 GPUs on CI (GH #111).")
def test_gmres_solve():
    N, D = 100, 100
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    # For mathematical reasons I don't understand, GMRES cannot converge
    # when solving this small system. To adequately test the solver, run
    # against scipy and compare the result. To make the test times relatively
    # small, we cap the max iteration count here. However, if both solvers
    # are allowed to run to completion, they match with the default small
    # error tolerance.
    x_pred_sci = scipy.sparse.linalg.gmres(
        A, y, atol=1e-5, tol=1e-5, maxiter=300
    )[0]
    x_pred_legate = linalg.gmres(A, y, atol=1e-5, tol=1e-5, maxiter=300)[0]
    assert np.allclose(x_pred_sci, x_pred_legate, atol=1e-1)


def test_eigsh():
    N = 100
    seed = 471014
    # Make A symmetric.
    A = sample(N, N, 0.1, seed).todense()
    A = 0.5 * (A + A.T)
    A = csr_array(A)
    vals_legate, vecs_legate = linalg.eigsh(A)
    # Check that all of the vectors are indeed equal to the eigenvalue.
    for i, lamb in enumerate(vals_legate):
        assert np.allclose(
            A @ vecs_legate[:, i], lamb * vecs_legate[:, i], atol=1e-3
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
