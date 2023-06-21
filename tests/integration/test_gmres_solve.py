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

import sparse.linalg as linalg
from sparse import csr_array, eye
from utils.sample import sample, sample_dense_vector


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
