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
import scipy.spatial.distance as sci_spatial

import sparse.spatial as legate_spatial


@pytest.mark.parametrize("ma", [20, 30, 40])
@pytest.mark.parametrize("mb", [20, 30, 40])
@pytest.mark.parametrize("n", [10, 20, 30])
def test_euclidean_cdist(ma, mb, n):
    XA = np.random.random((ma, n))
    XB = np.random.random((mb, n))
    arr = legate_spatial.cdist(XA, XB)
    s = sci_spatial.cdist(XA, XB)
    assert np.allclose(arr, s)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
