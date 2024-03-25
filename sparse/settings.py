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
#
from __future__ import annotations

from legate.util.settings import PrioritizedSetting, Settings, convert_bool

__all__ = ("settings",)


class SparseRuntimeSettings(Settings):
    precise_images: PrioritizedSetting[bool] = PrioritizedSetting(
        "precise-images",
        "LEGATE_SPARSE_PRECISE_IMAGES",
        default=False,
        convert=convert_bool,
        help="""
        Use precise images instead of approximate min-max boundary based
        images. This can potentially reduce communication volume at the cost of
        increasing startup time before application steady state.
        """,
    )


settings = SparseRuntimeSettings()
