# Copyright 2023 NVIDIA Corporation
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
import pytest
import scipy
from utils.sample import sample, sample_dense_vector

import sparse.linalg as linalg
from sparse import csr_array, eye


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
