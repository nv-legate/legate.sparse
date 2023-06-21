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
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_spmm(filename, b_type, c_type):
    arr = legate_io.mmread(filename).tocsr().astype(b_type)
    s = sci_io.mmread(filename).tocsr().astype(b_type)
    c = arr.todense().astype(c_type)
    res_legate = arr @ c
    res_sci = s @ c
    assert np.allclose(res_legate, res_sci)
    result = np.zeros(
        arr.shape, dtype=numpy.find_common_type([b_type, c_type], [])
    )
    arr.dot(c, out=result)
    assert np.allclose(res_legate, res_sci)


# We separate the tests parametrized over idim and types to avoid
# having the number of tests explode.
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
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_spmm_rmatmul_types(filename, b_type, c_type):
    arr = legate_io.mmread(filename).tocsr().astype(b_type)
    x = np.ones((8, arr.shape[1])).astype(c_type)
    s = sci_io.mmread(filename).tocsr().astype(b_type)
    # TODO (rohany): Until we have the dispatch with cunumeric
    #  then we can stop explicitly calling __rmatmul__. We also
    #  can't even do it against the scipy matrix because it doesn't
    #  have the overload either.
    assert np.allclose(arr.__rmatmul__(x), numpy.array(x) @ s)


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
