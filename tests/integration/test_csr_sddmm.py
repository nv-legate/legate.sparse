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
from legate.core.solver import Partitioner
from utils.common import test_mtx_files

import sparse.io as legate_io
from sparse import runtime


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
