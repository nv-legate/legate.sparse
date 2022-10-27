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
import pytest
import scipy.sparse as scpy

from sparse import eye, spdiags


def test_dia_to_csr():
    pass


def test_spdiags():
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    )
    diag = np.array([0, -1, 2])
    arr = spdiags(data, diag, 4, 4).todense()
    s = scpy.spdiags(data, diag, 4, 4).todense()
    assert np.array_equal(arr, s)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_eye(m, n, k):
    arr = eye(m, n=n, k=k, format="csr").todense()
    s = scpy.eye(m, n=n, k=k, format="csr").todense()
    assert np.array_equal(arr, s)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_dia_diagonal(m, n, k):
    arr = eye(m, n=n, k=k, format="dia")
    s = scpy.eye(m, n=n, k=k, format="dia")
    assert np.array_equal(arr.diagonal(k=k), s.diagonal(k=k))


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_dia_to_coo(m, n, k):
    arr = eye(m, n=n, k=k, format="dia").tocoo().todense()
    s = scpy.eye(m, n=n, k=k, format="dia").todense()
    assert np.array_equal(arr, s)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
