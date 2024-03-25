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
from utils.common import test_mtx_files, types

import sparse.io as legate_io


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_csr_csr_spgemm(filename, b_type, c_type):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename).tocsr()
    res_legate = arr.tocsr().astype(b_type) @ arr.tocsr().astype(c_type)
    res_sci = s.astype(b_type) @ s.astype(c_type)
    assert np.allclose(res_legate.todense(), res_sci.todense())


@pytest.mark.parametrize("filename", test_mtx_files)
@pytest.mark.parametrize("b_type", types)
@pytest.mark.parametrize("c_type", types)
def test_csr_csr_csc_spgemm(filename, b_type, c_type):
    arr = legate_io.mmread(filename)
    s = sci_io.mmread(filename)
    res_legate = arr.tocsr().astype(b_type) @ arr.tocsc().astype(c_type)
    res_sci = s.tocsr().astype(b_type) @ s.tocsc().astype(c_type)
    assert np.allclose(res_legate.todense(), res_sci.todense())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
