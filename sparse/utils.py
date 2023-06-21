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

import math
import traceback
from typing import Any

import cunumeric
import numpy
import pyarrow
from legate.core import Array, Future, Store, types

import sparse

from .config import SparseOpCode
from .runtime import ctx


# find_last_user_stacklevel gets the last stack frame index
# within legate sparse.
def find_last_user_stacklevel() -> int:
    stacklevel = 1
    for frame, _ in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("sparse"):
            break
        stacklevel += 1
    return stacklevel


# store_to_cunumeric_array converts a store to a cuNumeric array.
def store_to_cunumeric_array(store: Store):
    return cunumeric.asarray(store)


# get_store_from_cunumeric_array extracts a store from a cuNumeric array.
def get_store_from_cunumeric_array(
    arr: cunumeric.ndarray,
    allow_future=False,
    copy=False,
) -> Store:
    # TODO (rohany): It's unclear how to actually get stores from the
    # __legate_data_interface__ for cunumeric arrays. It seems to depend on
    # whether they are eager/deferred etc. Have to ask Mike and Wonchan about
    # this.
    #
    # data = arr.__legate_data_interface__["data"]
    # (_, array) = list(data.items())[0]
    # target = array.stores()[1]
    # if isinstance(target, Store):
    #     store = target
    # else:
    #     if isinstance(target, cunumeric.eager.EagerArray):
    #         target = target.to_deferred_array()
    #     store = target.base
    #
    # Because of https://github.com/nv-legate/cunumeric/issues/595, we can't
    # access stores of cunumeric arrays through the `__legate_data_interface__`
    # if the stores happen to have a complex type. So, we'll do something
    # hackier and just reach into the array's thunk and extract the store.
    if copy:
        # If requested to make a copy, do so.
        arr = cunumeric.array(arr)

    data = arr.__legate_data_interface__["data"]
    array = data[next(iter(data))]
    (_, store) = array.stores()

    # Our implementation can't handle future backed stores when we use this, as
    # we expect to be able to partition things up. If we have a future backed
    # store, create a normal store and issue a copy from the backed store to
    # the new store.
    if store.kind == Future and not allow_future:
        store_copy = ctx.create_store(store.type, shape=store.shape)
        task = ctx.create_auto_task(SparseOpCode.UPCAST_FUTURE_TO_REGION)
        task.add_output(store_copy)
        task.add_input(store)
        task.add_broadcast(store_copy)
        task.add_scalar_arg(store.type.size, types.uint64)
        task.execute()
        store = store_copy
    return store


# cast_to_store attempts to cast an arbitrary object into a store.
def cast_to_store(arr):
    if isinstance(arr, Store):
        return arr
    if isinstance(arr, numpy.ndarray):
        arr = cunumeric.array(arr)
    if isinstance(arr, cunumeric.ndarray):
        return get_store_from_cunumeric_array(arr)
    raise NotImplementedError


# cast_arr attempts to cast an arbitrary object into a cunumeric
# ndarray, with an optional desired type.
def cast_arr(arr, dtype=None):
    if isinstance(arr, Store):
        arr = store_to_cunumeric_array(arr)
    elif not isinstance(arr, cunumeric.ndarray):
        arr = cunumeric.array(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


# find_common_type performs a similar analysis to
# cunumeric.ndarray.find_common_type to find a common type
# between all of the arguments.
def find_common_type(*args):
    array_types = list()
    scalar_types = list()
    for array in args:
        if sparse.is_sparse_matrix(array):
            array_types.append(array.dtype)
        elif array.size == 1:
            scalar_types.append(array.dtype)
        else:
            array_types.append(array.dtype)
    return numpy.find_common_type(array_types, scalar_types)


# cast_to_common_type casts all arguments to the same common dtype.
def cast_to_common_type(*args):
    # Find a common type for all of the arguments.
    common_type = find_common_type(*args)
    # Cast each input to the common type. Ideally, if all of the
    # arguments are already the common type then this will
    # be a no-op.
    return tuple(arg.astype(common_type, copy=False) for arg in args)


# factor_int decomposes an integer into a close to square grid.
def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2


# broadcast_store broadcasts a store to the desired input shape,
# or throws an error if the broadcast is not possible.
def broadcast_store(store: Store, shape: Any) -> Store:
    diff = len(shape) - store.ndim
    for dim in range(diff):
        store = store.promote(dim, shape[dim])
    for dim in range(len(shape)):
        if store.shape[dim] != shape[dim]:
            if store.shape[dim] != 1:
                raise ValueError(
                    f"Shape did not match along dimension {dim} "
                    "and the value is not equal to 1"
                )
            store = store.project(dim, 0).promote(dim, shape[dim])
    return store
