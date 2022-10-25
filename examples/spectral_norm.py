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

# This spectral norm calculation was derived from https://github.com/pericycle/normest/.

import cunumeric as np
from sparse import csr_array

# Return an approximation of the 2 norm of a matrix.
def normest(M, tol=1e-4):
    # M is a symmetric positive semi-definite numpy array or a scipy sparse matrix
    # up to a tolerance of tol using the power method
    max_it = 10
    res = 1
    it_count = 0
    x = np.random.rand(M.shape[1], 1)
    y = M.dot(x)
    pnorm = np.sqrt(np.sum(y ** 2))
    x = y / pnorm
    while (res > tol) and (it_count < max_it):
        y = M.dot(x)
        ynorm = np.sqrt(np.sum(y ** 2))
        res = abs(pnorm - ynorm)
        pnorm = np.copy(ynorm)
        x = y / ynorm
        it_count += 1
    v = M.dot(x)
    return np.sqrt(np.sum(v ** 2))

np.random.seed(15210)
M = np.random.random((100, 100))
A = csr_array(M)
assert np.isclose(normest(A), normest(M))
