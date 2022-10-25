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

# This test file contains an assortment of tests for functions in the overall
# scipy.sparse module and arent tied to a particular format.
import cunumeric as np
import pytest
import sparse
import scipy.sparse as scpy

import sparse.io as legate_io
import scipy.io as sci_io
import numpy

from utils.common import test_mtx_files


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_kron(filename, format):
    l = legate_io.mmread(filename).asformat(format)
    s = sci_io.mmread(filename).asformat(format)
    res_legate = sparse.kron(l, sparse.coo_array(np.roll(l.todense(), 1)), format=format)
    res_sci = scpy.kron(s, np.roll(s.toarray(), 1), format=format)
    assert np.array_equal(res_legate.todense(), numpy.ascontiguousarray(res_sci.todense()))


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("format", ["coo", "csr", "csc"])
def test_diagonal(filename, k, format):
    l = legate_io.mmread(filename).asformat(format)
    s = sci_io.mmread(filename).asformat(format)
    # TODO (rohany): Convert this back.
    # if format != "coo" and k != 0:
    if k != 0:
        with pytest.raises(NotImplementedError):
            assert np.array_equal(l.diagonal(k), s.diagonal(k))
    else:
        assert np.array_equal(l.diagonal(k), s.diagonal(k))


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("format", ["csr"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum(filename, format, axis):
    l = legate_io.mmread(filename).asformat(format)
    s = sci_io.mmread(filename).asformat(format)
    res_legate = l.sum(axis=axis)
    res_sci = s.sum(axis=axis)
    if axis is None:
        assert np.allclose(res_legate, res_sci)
    else:
        # For some reason, scipy returns a 1xN 2-D matrix instead
        # of a vector here. To make the test work out, ravel the
        # resulting matrix into a vector.
        np.allclose(res_legate, np.array(res_sci).ravel())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv))
