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
import scipy.sparse as scpy
from utils.common import test_mtx_files, types

import sparse.io as legate_io
from sparse import csr_array


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_elemwise_mul(filename, b_type, c_type):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr().astype(b_type) * csr_array(
        np.roll(arr.todense(), 1)
    ).astype(c_type)
    res_scipy = s.tocsr().astype(b_type) * scpy.csr_array(
        np.roll(np.array(s.todense()), 1)
    ).astype(c_type)
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_dense_elemwise_mul(filename, b_type, c_type):
    arr = legate_io.mmread(filename).tocsr().astype(b_type)
    s = sci_io.mmread(filename).tocsr().astype(b_type)
    c = np.random.random(arr.shape).astype(c_type)
    res_legate = arr * c
    # The version of scipy that I have locally still thinks * is matmul.
    res_scipy = s.multiply(numpy.array(c))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_elemwise_add(filename, b_type, c_type):
    arr = legate_io.mmread(filename).tocsr().astype(b_type)
    s = sci_io.mmread(filename).tocsr().astype(b_type)
    res_legate = arr + csr_array(np.roll(arr.todense().astype(c_type), 1))
    res_scipy = s + scpy.csr_array(np.roll(s.toarray().astype(c_type), 1))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_mul_scalar(filename):
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    res_legate = arr * 3.0
    res_sci = s * 3.0
    assert np.allclose(res_legate.todense(), res_sci.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_subtract(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr() - csr_array(np.roll(arr.todense(), 1))
    res_scipy = s.tocsr() - scpy.csr_array(np.roll(s.toarray(), 1))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


def test_csr_power():
    filename = "testdata/test.mtx"
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    assert np.allclose(arr.power(2).todense(), s.power(2).todense())


def test_csr_neg():
    filename = "testdata/test.mtx"
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    assert np.allclose((-arr).todense(), (-s).todense())


def test_mult_dense_broadcast():
    filename = "testdata/test.mtx"
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    x = np.ones(arr.shape[0])
    # TODO (rohany): I don't know why I need to cast
    #  x into a numpy array from a cunumeric array.
    assert np.allclose(
        arr.multiply(x).todense(), s.multiply(numpy.array(x)).todense()
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
