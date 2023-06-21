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
import scipy.io as sci_io
import scipy.sparse as scpy
from utils.common import test_mtx_files, types

import sparse.io as legate_io
from sparse import csr_array


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_from_dense(filename):
    arr = csr_array(legate_io.mmread(filename).todense())
    s = scpy.csr_array(sci_io.mmread(filename).todense())
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_to_coo(filename):
    arr = legate_io.mmread(filename).tocsr()
    assert np.array_equal(arr.todense(), arr.tocoo().todense())


def test_csr_coo_constructor():
    f = test_mtx_files[0]
    coo = legate_io.mmread(f).tocoo()
    csr = csr_array(
        (coo.data, (coo.row, coo.col)), dtype=coo.dtype, shape=coo.shape
    )
    assert np.array_equal(coo.todense(), csr.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_from_scipy_csr(filename):
    s = scpy.csr_array(sci_io.mmread(filename).todense()).astype(np.float64)
    arr = csr_array(s)
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("copy", [False])
def test_csr_conj(filename, copy):
    arr = legate_io.mmread(filename).tocsr().conj(copy=copy)
    s = sci_io.mmread(filename).tocsr().conj(copy=copy)
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_to_scipy_csr(filename):
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    assert np.array_equal(arr.to_scipy_sparse_csr().todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("dtype", types)
def test_csr_todense(filename, dtype):
    arr = legate_io.mmread(filename).tocsr().astype(dtype)
    s = sci_io.mmread(filename).tocsr().astype(dtype)
    assert np.array_equal(arr.todense(), s.todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
