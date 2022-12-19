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

#include "sparse/array/conv/csc_to_dense.h"
#include "sparse/array/conv/csc_to_dense_template.inl"
#include "sparse/util/cusparse_utils.h"

namespace sparse {

using namespace Legion;
using namespace legate;

template <typename INDEX_TY, typename VAL_TY>
__global__ void CSCtoDenseKernel(size_t cols,
                                 Rect<2> bounds,
                                 AccessorWO<VAL_TY, 2> A_vals,
                                 AccessorRO<Rect<1>, 1> B_pos,
                                 AccessorRO<INDEX_TY, 1> B_crd,
                                 AccessorRO<VAL_TY, 1> B_vals)
{
  const auto idx = global_tid_1d();
  if (idx >= cols) return;
  INDEX_TY j = idx + bounds.lo[1];
  // Initialize the row with all zeros.
  for (INDEX_TY i = bounds.lo[0]; i < bounds.hi[0] + 1; i++) { A_vals[{i, j}] = 0.0; }
  // Copy the non-zero values into place.
  for (INDEX_TY iB = B_pos[j].lo; iB < B_pos[j].hi + 1; iB++) {
    INDEX_TY i     = B_crd[iB];
    A_vals[{i, j}] = B_vals[iB];
  }
}

template <>
struct CSCToDenseImpl<VariantKind::GPU> {
  template <LegateTypeCode INDEX_CODE, LegateTypeCode VAL_CODE>
  void operator()(CSCToDenseArgs& args) const
  {
    using INDEX_TY = legate_type_of<INDEX_CODE>;
    using VAL_TY   = legate_type_of<VAL_CODE>;

    auto& A_vals = args.A_vals;
    auto& B_pos  = args.B_pos;
    auto& B_crd  = args.B_crd;
    auto& B_vals = args.B_vals;

    // Break out early if the iteration space partition is empty.
    if (B_pos.domain().empty()) { return; }

    auto stream = get_cached_stream();

    // If we are running on an old cuSPARSE version, then we don't
    // have access to many cuSPARSE functions. In that case, use
    // a hand-written version.
#if (CUSPARSE_VER_MAJOR < 11 || (CUSPARSE_VER_MAJOR == 11 && CUSPARSE_VER_MINOR < 2))
    auto B_domain = B_pos.domain();
    auto cols     = B_domain.hi()[0] - B_domain.lo()[0] + 1;
    auto blocks   = get_num_blocks_1d(cols);
    CSCtoDenseKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(cols,
                                                               A_vals.shape<2>(),
                                                               A_vals.write_accessor<VAL_TY, 2>(),
                                                               B_pos.read_accessor<Rect<1>, 1>(),
                                                               B_crd.read_accessor<INDEX_TY, 1>(),
                                                               B_vals.read_accessor<VAL_TY, 1>());
#else
    // Get context sensitive objects.
    auto handle = get_cusparse();
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Construct our cuSPARSE matrices.
    auto A_domain   = A_vals.domain();
    auto cusparse_A = makeCuSparseDenseMat<VAL_TY>(A_vals);
    auto cusparse_B = makeCuSparseCSC<INDEX_TY, VAL_TY>(
      B_pos, B_crd, B_vals, A_domain.hi()[0] - A_domain.lo()[0] + 1 /* rows */);

    // Finally make the cuSPARSE calls.
    size_t bufSize = 0;
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
      handle, cusparse_B, cusparse_A, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufSize));
    // Allocate a buffer if we need to.
    void* workspacePtr = nullptr;
    if (bufSize > 0) {
      DeferredBuffer<char, 1> buf({0, bufSize - 1}, Memory::GPU_FB_MEM);
      workspacePtr = buf.ptr(0);
    }
    // Finally do the conversion.
    CHECK_CUSPARSE(cusparseSparseToDense(
      handle, cusparse_B, cusparse_A, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workspacePtr));
    // Destroy the created objects.
    CHECK_CUSPARSE(cusparseDestroyDnMat(cusparse_A));
    CHECK_CUSPARSE(cusparseDestroySpMat(cusparse_B));
#endif

    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void CSCToDense::gpu_variant(TaskContext& context)
{
  csc_to_dense_template<VariantKind::GPU>(context);
}

}  // namespace sparse
