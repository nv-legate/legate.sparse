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
from sparse import csc_array, runtime
import scipy.sparse as scpy
from legate.core.solver import Partitioner
import os

import sparse.io as legate_io
import scipy.io as sci_io
import numpy

from common import test_mtx_files


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_from_dense(filename):
    l = csc_array(legate_io.mmread(filename).todense())
    s = scpy.csc_array(sci_io.mmread(filename).todense())
    assert np.array_equal(l.todense(), s.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_to_coo(filename):
    l = legate_io.mmread(filename).tocsc()
    assert np.array_equal(l.todense(), l.tocoo().todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_to_csr(filename):
    l = legate_io.mmread(filename).tocsc()
    assert np.array_equal(l.tocsr().todense(), l.tocoo().todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_elemwise_mul(filename):
    l = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = l.tocsc() * csc_array(np.roll(l.todense(), 1))
    res_scipy = s.tocsc() * scpy.csc_array(np.roll(np.array(s.todense()), 1))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_elemwise_add(filename):
    l = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = l.tocsc() + csc_array(np.roll(l.todense(), 1))
    # Annoyingly, it looks like we have to cast the returned scipy dense
    # array into a numpy array or some wonky things happen...
    res_scipy = s.tocsc() + scpy.csc_array(np.roll(np.array(s.todense()), 1))
    assert np.allclose(res_legate.todense(), numpy.ascontiguousarray(res_scipy.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_transpose(filename):
    l = legate_io.mmread(filename).tocsc().T
    s = sci_io.mmread(filename).tocsc().T
    assert np.array_equal(l.todense(), numpy.ascontiguousarray(s.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("copy", [False])
def test_csc_conj(filename, copy):
    l = legate_io.mmread(filename).tocsc().conj(copy=copy)
    s = sci_io.mmread(filename).tocsc().conj(copy=copy)
    assert np.array_equal(l.todense(), numpy.ascontiguousarray(s.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_dot(filename):
    # Test vectors and n-1 matrices.
    l = legate_io.mmread(filename).tocsc()
    s = sci_io.mmread(filename).tocsc()
    vec = np.random.random((l.shape[0]))
    assert np.allclose(l @ vec, s @ vec)
    assert np.allclose(l.dot(vec), s.dot(vec))
    result_l = np.zeros((l.shape[0]))
    l.dot(vec, out=result_l)
    assert np.allclose(result_l, s.dot(vec))
    vec = np.random.random((l.shape[0], 1))
    assert np.allclose(l @ vec, s @ vec)
    assert np.allclose(l.dot(vec), s.dot(vec))
    result_l = np.zeros((l.shape[0], 1))
    l.dot(vec, out=result_l)
    assert np.allclose(result_l, s.dot(vec))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_dot_mat(filename):
    l = legate_io.mmread(filename).tocsc()
    s = sci_io.mmread(filename).tocsc()
    mat = np.random.random(l.shape)
    assert np.allclose(l.dot(mat), s.dot(mat))


@pytest.mark.parametrize("filename", test_mtx_files)
def test_csc_todense(filename):
    l = legate_io.mmread(filename).tocsc()
    s = sci_io.mmread(filename).tocsc()
    assert np.array_equal(l.todense(), numpy.ascontiguousarray(s.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("kdim", [2, 4, 8, 16])
def test_csc_sddmm(filename, kdim):
    l = legate_io.mmread(filename).tocsc()
    s = sci_io.mmread(filename).tocsc()
    C = np.random.random((l.shape[0], kdim))
    D = np.random.random((kdim, l.shape[1]))
    res_legate = l.sddmm(C, D)
    # This version of scipy still thinks that * is matrix multiplication instead
    # of element-wise multiplication, so we have to use multiply().
    res_scipy = s.multiply(numpy.array(C @ D))
    assert np.allclose(res_legate.todense(), res_scipy.todense())


def test_csc_sddmm_balanced():
    sparse_rt = runtime.runtime
    rt = sparse_rt.legate_runtime
    if sparse_rt.num_procs == 1:
        pytest.skip("Must run with multiple processors.")
    if "LEGATE_TEST" not in os.environ:
        pytest.skip("Partitioning must be forced with LEGATE_TEST=1")
    filename = "testdata/test.mtx"
    l = legate_io.mmread(filename).tocsc()
    s = sci_io.mmread(filename).tocsc()
    l.balance()
    # idim must be small enough so that the solver doesn't think that
    # re-partitioning x or the output will cause more data movement.
    kdim = 2
    C = np.random.random((l.shape[0], kdim))
    D = np.random.random((kdim, l.shape[1]))
    rt._window_size = 2
    rt.flush_scheduling_window()
    res = l.sddmm(C, D)
    assert len(rt._outstanding_ops) == 1
    partitioner = Partitioner(rt, [rt._outstanding_ops[0]], must_be_single=False)
    strat = partitioner.partition_stores()
    assert "by_domain" in str(strat)
    rt._window_size = 1
    rt.flush_scheduling_window()
    # Ensure that the answer is correct.
    assert np.allclose(res.todense(), s.multiply(numpy.array(C @ D)).todense())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv))
