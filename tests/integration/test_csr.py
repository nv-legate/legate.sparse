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

import os

import cunumeric as np
import numpy
import pytest
import scipy.io as sci_io
import scipy.sparse as scpy
from legate.core.solver import Partitioner
from utils.common import test_mtx_files, types

import sparse.io as legate_io
from sparse import csr_array, runtime


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
@pytest.mark.parametrize("mat_type", types)
@pytest.mark.parametrize("vec_type", types)
def test_csr_dot(filename, mat_type, vec_type):
    # Test vectors and n-1 matrices.
    arr = legate_io.mmread(filename).tocsr().astype(mat_type)
    s = sci_io.mmread(filename).tocsr().astype(mat_type)
    vec = np.random.random((arr.shape[0])).astype(vec_type)
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec), s.dot(vec))
    out_type = numpy.find_common_type([mat_type, vec_type], [])
    result_l = np.zeros((arr.shape[0]), dtype=out_type)
    arr.dot(vec, out=result_l)
    assert np.allclose(result_l, s.dot(vec))
    vec = np.random.random((arr.shape[0], 1)).astype(vec_type)
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec), s.dot(vec))
    result_l = np.zeros((arr.shape[0], 1), dtype=out_type)
    arr.dot(vec, out=result_l)
    assert np.allclose(result_l, s.dot(vec))


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
def test_csr_spmm(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr() @ arr.todense()
    res_sci = s.tocsr() @ s.todense()
    assert np.allclose(res_legate, res_sci)
    result = np.zeros(arr.shape)
    arr.tocsr().dot(arr.todense(), out=result)
    assert np.allclose(res_legate, res_sci)


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("idim", [2, 4, 8, 16])
def test_csr_spmm_rmatmul(filename, idim):
    arr = legate_io.mmread(filename).tocsr()
    x = np.ones((idim, arr.shape[1]))
    s = sci_io.mmread(filename).tocsr()
    # TODO (rohany): Until we have the dispatch with cunumeric
    #  then we can stop explicitly calling __rmatmul__. We also
    #  can't even do it against the scipy matrix because it doesn't
    #  have the overload either.
    assert np.allclose(arr.__rmatmul__(x), numpy.array(x) @ s)


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_csr_csr_spgemm(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename).tocsr()
    res_legate = arr.tocsr() @ arr.tocsr()
    res_sci = s @ s
    assert np.allclose(res_legate.todense(), res_sci.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_csr_csc_spgemm(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr() @ arr.tocsc()
    res_sci = s.tocsr() @ s.tocsc()
    assert np.allclose(res_legate.todense(), res_sci.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_transpose(filename):
    arr = legate_io.mmread(filename).tocsr().T
    s = sci_io.mmread(filename).tocsr().T
    assert np.array_equal(arr.todense(), numpy.ascontiguousarray(s.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_todense(filename):
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    assert np.array_equal(arr.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_elemwise_mul(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr() * csr_array(np.roll(arr.todense(), 1))
    res_scipy = s.tocsr() * scpy.csr_array(np.roll(np.array(s.todense()), 1))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("kdim", [2, 4, 8, 16])
def test_csr_sddmm(filename, kdim):
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    C = np.random.random((arr.shape[0], kdim))
    D = np.random.random((kdim, arr.shape[1]))
    res_legate = arr.sddmm(C, D)
    # This version of scipy still thinks that * is matrix multiplication
    # instead of element-wise multiplication, so we have to use multiply().
    res_scipy = s.multiply(numpy.array(C @ D))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_dense_elemwise_mul(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    c = np.random.random(arr.shape)
    res_legate = arr.tocsr() * c
    # The version of scipy that I have locally still thinks * is matmul.
    res_scipy = s.tocsr().multiply(numpy.array(c))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_elemwise_add(filename):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr() + csr_array(np.roll(arr.todense(), 1))
    res_scipy = s.tocsr() + scpy.csr_array(np.roll(s.toarray(), 1))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csr_to_scipy_csr(filename):
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    assert np.array_equal(arr.to_scipy_sparse_csr().todense(), s.todense())


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


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("k", [0])
def test_csr_diagonal(filename, k):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr().diagonal(k=k)
    res_sci = s.tocsr().diagonal(k=k)
    assert np.array_equal(res_legate, res_sci)


# The goal of this test is to ensure that when we balance the rows
# of CSR matrix, we actually use that partition in operations. We'll
# do this by by being really hacky. We'll increase the runtime's window
# size, and inspecting what would happen if the solver partitions the
# operation within the window.
def test_csr_rmatmul_balanced():
    sparse_rt = runtime.runtime
    rt = sparse_rt.legate_runtime
    if sparse_rt.num_procs == 1:
        pytest.skip("Must run with multiple processors.")
    if "LEGATE_TEST" not in os.environ:
        pytest.skip("Partitioning must be forced with LEGATE_TEST=1")
    filename = "testdata/test.mtx"
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    arr.balance()
    # idim must be small enough so that the solver doesn't think that
    # re-partitioning x or the output will cause more data movement.
    idim = 2
    x = np.ones((idim, arr.shape[1]))
    rt._window_size = 3
    rt.flush_scheduling_window()
    res = arr.__rmatmul__(x)
    # We expect to find the cunumeric zero task and the SpMM task.
    assert len(rt._outstanding_ops) == 2
    partitioner = Partitioner([rt._outstanding_ops[1]], must_be_single=False)
    strat = partitioner.partition_stores()
    assert "by_domain" in str(strat)
    rt._window_size = 1
    rt.flush_scheduling_window()
    # Ensure that the answer is correct.
    assert np.allclose(res, numpy.array(x) @ s)


def test_csr_sddmm_balanced():
    sparse_rt = runtime.runtime
    rt = sparse_rt.legate_runtime
    if sparse_rt.num_procs == 1:
        pytest.skip("Must run with multiple processors.")
    if "LEGATE_TEST" not in os.environ:
        pytest.skip("Partitioning must be forced with LEGATE_TEST=1")
    filename = "testdata/test.mtx"
    arr = legate_io.mmread(filename).tocsr()
    s = sci_io.mmread(filename).tocsr()
    arr.balance()
    # idim must be small enough so that the solver doesn't think that
    # re-partitioning x or the output will cause more data movement.
    kdim = 2
    C = np.random.random((arr.shape[0], kdim))
    D = np.random.random((kdim, arr.shape[1]))
    rt._window_size = 2
    rt.flush_scheduling_window()
    res = arr.sddmm(C, D)
    assert len(rt._outstanding_ops) == 1
    partitioner = Partitioner([rt._outstanding_ops[0]], must_be_single=False)
    strat = partitioner.partition_stores()
    assert "by_domain" in str(strat)
    rt._window_size = 1
    rt.flush_scheduling_window()
    # Ensure that the answer is correct.
    assert np.allclose(res.todense(), s.multiply(numpy.array(C @ D)).todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
