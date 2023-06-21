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
@pytest.mark.parametrize("mat_type", types)
@pytest.mark.parametrize("vec_type", types)
@pytest.mark.parametrize("col_split", [True, False])
def test_csr_dot(filename, mat_type, vec_type, col_split):
    # Test vectors and n-1 matrices.
    arr = legate_io.mmread(filename).tocsr().astype(mat_type)
    s = sci_io.mmread(filename).tocsr().astype(mat_type)
    vec = np.random.random((arr.shape[0])).astype(vec_type)
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec, spmv_domain_part=col_split), s.dot(vec))
    out_type = numpy.find_common_type([mat_type, vec_type], [])
    result_l = np.zeros((arr.shape[0]), dtype=out_type)
    arr.dot(vec, spmv_domain_part=col_split, out=result_l)
    assert np.allclose(result_l, s.dot(vec))
    vec = np.random.random((arr.shape[0], 1)).astype(vec_type)
    assert np.allclose(arr @ vec, s @ vec)
    assert np.allclose(arr.dot(vec), s.dot(vec))
    result_l = np.zeros((arr.shape[0], 1), dtype=out_type)
    arr.dot(vec, out=result_l)
    assert np.allclose(result_l, s.dot(vec))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
