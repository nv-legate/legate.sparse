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
import scipy.io as sci_io
import scipy.sparse as scpy
from utils.common import test_mtx_files, types

import sparse.io as legate_io
from sparse import coo_array


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("dtype", types)
def test_coo_from_scipy(filename, dtype):
    s = scpy.coo_array(sci_io.mmread(filename).todense()).astype(dtype)
    arr = coo_array(s)
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_coo_from_arrays(filename):
    s = scpy.coo_array(sci_io.mmread(filename).todense())
    data, row, col = (
        np.array(s.data, np.float64),
        np.array(s.row, np.int64),
        np.array(s.col, np.int64),
    )
    arr = coo_array((data, (row, col)), shape=s.shape)
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_coo_transpose(filename):
    arr = legate_io.mmread(filename).tocoo()
    s = sci_io.mmread(filename).tocoo()
    assert np.array_equal(arr.T.todense(), s.T.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_coo_matmul(filename):
    arr = legate_io.mmread(filename).tocoo()
    s = sci_io.mmread(filename).tocoo()
    res_legate = arr @ arr.todense()
    res_sci = s @ s.todense()
    assert np.allclose(res_legate, numpy.ascontiguousarray(res_sci))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_coo_mul(filename):
    arr = legate_io.mmread(filename).tocoo()
    s = sci_io.mmread(filename).tocoo()
    res_legate = arr * 3.0
    res_sci = s * 3.0
    assert np.allclose(res_legate.todense(), res_sci.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_coo_dot(filename):
    arr = legate_io.mmread(filename).tocoo()
    s = sci_io.mmread(filename).tocoo()
    res_legate = arr.dot(arr.todense())
    res_sci = s.dot(s.todense())
    assert np.allclose(res_legate, numpy.ascontiguousarray(res_sci))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
