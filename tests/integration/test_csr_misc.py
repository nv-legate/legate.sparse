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
import numpy
import pytest
import scipy.io as sci_io
from utils.common import test_mtx_files

import sparse.io as legate_io


@pytest.mark.parametrize("filename", test_mtx_files)
def test_balance_row_partitions(filename):
    # Test vectors and n-1 matrices.
    arr = legate_io.mmread(filename).tocsr()
    arr.balance()
    s = sci_io.mmread(filename).tocsr()
    vec = np.random.random((arr.shape[0]))
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec), s.dot(vec))
    vec = np.random.random((arr.shape[0], 1))
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec), s.dot(vec))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_transpose(filename):
    arr = legate_io.mmread(filename).tocsr().T
    s = sci_io.mmread(filename).tocsr().T
    assert np.array_equal(arr.todense(), numpy.ascontiguousarray(s.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("k", [0])
def test_csr_diagonal(filename, k):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr().diagonal(k=k)
    res_sci = s.tocsr().diagonal(k=k)
    assert np.array_equal(res_legate, res_sci)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
