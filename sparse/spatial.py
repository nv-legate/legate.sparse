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

from .runtime import ctx, runtime
from .array import get_store_from_cunumeric_array, store_to_cunumeric_array, float64, factor_int
from .config import SparseOpCode

import cunumeric
import numpy

from legate.core import Rect
from legate.core.shape import Shape
from legate.core.partition import Tiling


# TODO (rohany): This function is not technically part of scipy.sparse, but I need it
#  for another application...
def cdist(XA, XB, metric='euclidean', *, out=None, **kwargs):
    if metric != 'euclidean':
        raise NotImplementedError
    if isinstance(XA, numpy.ndarray):
        XA = cunumeric.ndarray(XA)
    if isinstance(XB, numpy.ndarray):
        XB = cunumeric.ndarray(XB)
    assert len(XA.shape) == 2 and len(XB.shape) == 2 and XA.shape[1] == XB.shape[1]

    XA_store = get_store_from_cunumeric_array(XA)
    XB_store = get_store_from_cunumeric_array(XB)
    # We'll launch over a 2-D grid, parallelizing over tiles of the
    # output matrix.
    output = ctx.create_store(float64, shape=(XA.shape[0], XB.shape[0]))
    num_procs = runtime.num_procs
    grid = Shape(factor_int(num_procs))
    x_tiling = (output.shape[0] + grid[0] - 1) // grid[0]
    y_tiling = (output.shape[1] + grid[1] - 1) // grid[1]

    task = ctx.create_manual_task(SparseOpCode.EUCLIDEAN_CDIST, launch_domain=Rect(hi=grid))
    # First, we partition the output into tiles.
    task.add_output(output.partition(Tiling(Shape((x_tiling, y_tiling)), grid)))
    # For each output tile, we need the corresponding set of rows in XA. We'll do this
    # creating a disjoint partition over just the grid[0] color space, and adding a
    # projection functor on the launch space to pick corresponding piece within the rows.
    task.add_input(XA_store.partition(Tiling(Shape((x_tiling, XA.shape[1])), Shape((grid[0], 1)))), proj=lambda p: (p[0], 0))
    # We do a similar thing for XB, but this time we'll take the projection of the columns
    # of the processor grid.
    task.add_input(XB_store.partition(Tiling(Shape((y_tiling, XB_store.shape[1])), Shape((grid[1], 1)))), proj=lambda p: (p[1], 0))
    task.execute()
    return store_to_cunumeric_array(output)

