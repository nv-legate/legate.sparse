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
from utils.sample import sample

import sparse.linalg as linalg
from sparse import csr_array


@pytest.mark.xfail(reason="Seems to be failing on 2 procs on CI (GH #114).")
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
