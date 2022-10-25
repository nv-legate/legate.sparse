/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "cuda_help.h"

namespace sparse {

struct CUDALibraries {
public:
  CUDALibraries();
  ~CUDALibraries();

private:
  // Prevent copying and overwriting.
  CUDALibraries(const CUDALibraries& rhs)            = delete;
  CUDALibraries& operator=(const CUDALibraries& rhs) = delete;

public:
  void finalize();
  cusparseHandle_t get_cusparse();

private:
  void finalize_cusparse();

private:
  bool finalized_;
  cusparseHandle_t cusparse_;
};

}  // namespace sparse
