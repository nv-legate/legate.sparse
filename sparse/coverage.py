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
#
from __future__ import annotations

from functools import wraps
from types import FunctionType, MethodDescriptorType, MethodType, ModuleType
from typing import Any, Container, Mapping, Optional, cast

from legate.core import track_provenance
from typing_extensions import Protocol

MOD_INTERNAL = {"__dir__", "__getattr__"}


def filter_namespace(
    ns: Mapping[str, Any],
    *,
    omit_names: Optional[Container[str]] = None,
    omit_types: tuple[type, ...] = (),
) -> dict[str, Any]:
    omit_names = omit_names or set()
    return {
        attr: value
        for attr, value in ns.items()
        if attr not in omit_names and not isinstance(value, omit_types)
    }


def should_wrap(obj: object) -> bool:
    return isinstance(obj, (FunctionType, MethodType, MethodDescriptorType))


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


def wrap(func: AnyCallable) -> Any:
    @wraps(func)
    @track_provenance(nested=True)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def clone_module(
    origin_module: ModuleType, new_globals: dict[str, Any]
) -> None:
    """Copy attributes from one module to another, excluding submodules

    Function types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    Parameters
    ----------
    origin_module : ModuleTpe
        Existing module to clone attributes from

    new_globals : dict
        a globals() dict for the new module to clone into

    Returns
    -------
    None

    """
    for attr, value in new_globals.items():
        # Only need to wrap things that are in the origin module to begin with
        if attr not in origin_module.__dict__:
            continue
        if isinstance(value, FunctionType):
            wrapped = wrap(cast(AnyCallable, value))
            new_globals[attr] = wrapped


def clone_scipy_arr_kind(origin_class: type) -> Any:
    """Copy attributes from an origin class to the input class.

    Method types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    """

    def body(cls: type):
        for attr, value in cls.__dict__.items():
            # Only need to wrap things that are in the origin class to begin
            # with
            if not hasattr(origin_class, attr):
                continue
            if should_wrap(value):
                wrapped = wrap(value)
                setattr(cls, attr, wrapped)

        return cls

    return body
