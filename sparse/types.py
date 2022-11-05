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

import numpy

# Define some common types. Hopefully as we make more
# progress in generalizing the compute kernels, we can
# remove this code.
coord_ty = numpy.dtype(numpy.int64)
nnz_ty = numpy.dtype(numpy.uint64)
float64 = numpy.dtype(numpy.float64)
int32 = numpy.dtype(numpy.int32)
int64 = numpy.dtype(numpy.int64)
uint64 = numpy.dtype(numpy.uint64)
