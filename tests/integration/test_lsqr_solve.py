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
from utils.sample import sample, sample_dense_vector

import sparse.linalg as linalg
from sparse import csr_array


def test_lsqr_solve():
    N, D = 1000, 500
    seed = 471014
    A = csr_array(sample(N, D, 0.1, seed).todense())
    x = sample_dense_vector(D, 0.1, seed)
    y = A @ x
    assert np.allclose((A @ x), y)
    x_pred = linalg.lsqr(A, y, atol=1e-10, btol=1e-10)[0]
    assert np.allclose((A @ x_pred), y)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
