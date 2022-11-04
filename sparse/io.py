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

from legate.core import Future, track_provenance, types
from legate.core.shape import Shape

from .config import SparseOpCode
from .coo import coo_array
from .runtime import ctx, runtime
from .types import coord_ty, float64, nnz_ty


@track_provenance(runtime.legate_context)
def mmread(source):
    # TODO (rohany): We'll assume for now that all of the nodes in the system
    # can access the file passed in, so we don't need to worry about where this
    # task gets mapped to.
    rows = ctx.create_store(coord_ty, ndim=1)
    cols = ctx.create_store(coord_ty, ndim=1)
    vals = ctx.create_store(float64, ndim=1)
    m = ctx.create_store(coord_ty, optimize_scalar=True, shape=Shape(1))
    n = ctx.create_store(coord_ty, optimize_scalar=True, shape=Shape(1))
    nnz = ctx.create_store(nnz_ty, optimize_scalar=True, shape=Shape(1))
    assert m.kind == Future
    assert n.kind == Future
    task = ctx.create_task(SparseOpCode.READ_MTX_TO_COO)
    task.add_output(rows)
    task.add_output(cols)
    task.add_output(vals)
    task.add_output(m)
    task.add_output(n)
    task.add_output(nnz)
    task.add_scalar_arg(source, types.string)
    task.execute()
    m = int.from_bytes(m.storage.get_buffer(), "little")
    n = int.from_bytes(n.storage.get_buffer(), "little")
    nnz = int.from_bytes(nnz.storage.get_buffer(), "little")
    # Slice down each store from the resulting size into the actual size.
    sl = slice(0, nnz)
    rows, cols, vals = rows.slice(0, sl), cols.slice(0, sl), vals.slice(0, sl)
    return coo_array((vals, (rows, cols)), shape=(m, n))
