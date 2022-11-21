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

#include "sparse.h"
#include "cuda_help.h"

namespace sparse {

// All of our indices are 0 based.
const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

// convertGlobalPosToLocalIndPtr converts the global pos array used for
// CSR and CSC matrices into a locally indexed indptr array.
template <typename T>
__global__ void convertGlobalPosToLocalIndPtr(size_t rows, const Legion::Rect<1>* pos, T* indptr)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  // Offset each entry in pos down to the first element in pos to locally
  // index this piece of the CSR array.
  indptr[idx] = T(pos[idx].lo - pos[0].lo);
  // We also need to fill in the final rows+1 index of indptr to be the
  // total number of non-zeros. We'll have the first thread do this.
  if (idx == 0) { indptr[rows] = T(pos[rows - 1].hi + 1 - pos[0].lo); }
}

// getPtrFromStore is a utility to extract a pointer of the right type
// from a store.
template <typename T, int DIM>
void* getPtrFromStore(const legate::Store& store)
{
  auto dom = store.domain();
  if (store.is_writable() && store.is_readable()) {
    return store.read_write_accessor<T, DIM>().ptr(dom.lo());
  } else if (store.is_writable() && !store.is_readable()) {
    return store.write_accessor<T, DIM>().ptr(dom.lo());
  } else if (!store.is_writable() && store.is_readable()) {
    return const_cast<T*>(store.read_accessor<T, DIM>().ptr(dom.lo()));
  } else if (store.is_reducible()) {
    return store.reduce_accessor<Legion::SumReduction<T>, true /* exclusive */, DIM>().ptr(
      dom.lo());
  } else {
    assert(false);
    return nullptr;
  }
}

// Template dispatch for value type.
template <typename VAL_TY>
cudaDataType cusparseDataType();

template <>
inline cudaDataType cusparseDataType<float>()
{
  return CUDA_R_32F;
}

template <>
inline cudaDataType cusparseDataType<double>()
{
  return CUDA_R_64F;
}

template <>
inline cudaDataType cusparseDataType<complex<float>>()
{
  return CUDA_C_32F;
}

template <>
inline cudaDataType cusparseDataType<complex<double>>()
{
  return CUDA_C_64F;
}

// Template dispatch for the index type.
template <typename INDEX_TY>
cusparseIndexType_t cusparseIndexType();

// TODO (rohany): Can we handle unsigned integers?
template <>
inline cusparseIndexType_t cusparseIndexType<int32_t>()
{
  return CUSPARSE_INDEX_32I;
}

template <>
inline cusparseIndexType_t cusparseIndexType<int64_t>()
{
  return CUSPARSE_INDEX_64I;
}

// makeCuSparseCSR creates a cusparse CSR matrix from input arrays.
template <typename INDEX_TY = int64_t, typename VAL_TY = double>
cusparseSpMatDescr_t makeCuSparseCSR(const legate::Store& pos,
                                     const legate::Store& crd,
                                     const legate::Store& vals,
                                     size_t cols)
{
  cusparseSpMatDescr_t matDescr;
  auto stream = get_cached_stream();

  auto pos_domain = pos.domain();
  auto crd_domain = crd.domain();

  auto pos_acc = pos.read_accessor<Legion::Rect<1>, 1>();
  size_t rows  = pos_domain.get_volume();
  // TODO (rohany): In the future, we could allow for control over what integer
  //  type is used to represent the indptr array.
  Legion::DeferredBuffer<int64_t, 1> indptr({0, rows}, Legion::Memory::GPU_FB_MEM);
  auto blocks = get_num_blocks_1d(rows);
  convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    rows, pos_acc.ptr(pos_domain.lo()), indptr.ptr(0));

  CHECK_CUSPARSE(
    cusparseCreateCsr(&matDescr,
                      rows,
                      cols,
                      crd_domain.get_volume() /* nnz */,
                      (void*)indptr.ptr(0),
                      crd_domain.empty() ? nullptr : getPtrFromStore<INDEX_TY, 1>(crd),
                      vals.domain().empty() ? nullptr : getPtrFromStore<VAL_TY, 1>(vals),
                      cusparseIndexType<int64_t>(),
                      cusparseIndexType<INDEX_TY>(),
                      index_base,
                      cusparseDataType<VAL_TY>()));

  return matDescr;
}

// makeCuSparseCSC creates a cusparse CSC matrix from input arrays.
template <typename INDEX_TY = int64_t, typename VAL_TY = double>
cusparseSpMatDescr_t makeCuSparseCSC(const legate::Store& pos,
                                     const legate::Store& crd,
                                     const legate::Store& vals,
                                     size_t rows)
{
  cusparseSpMatDescr_t matDescr;
  auto stream = get_cached_stream();

  auto pos_domain = pos.domain();
  auto crd_domain = crd.domain();

  auto pos_acc = pos.read_accessor<Legion::Rect<1>, 1>();
  size_t cols  = pos_domain.get_volume();
  Legion::DeferredBuffer<int64_t, 1> indptr({0, cols}, Legion::Memory::GPU_FB_MEM);
  auto blocks = get_num_blocks_1d(cols);
  convertGlobalPosToLocalIndPtr<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    cols, pos_acc.ptr(pos_domain.lo()), indptr.ptr(0));

#if (CUSPARSE_VER_MAJOR < 11 || CUSPARSE_VER_MINOR < 2)
  assert(false && "cuSPARSE version too old! Try later than 11.1.");
#else
  CHECK_CUSPARSE(
    cusparseCreateCsc(&matDescr,
                      rows,
                      cols,
                      crd_domain.get_volume() /* nnz */,
                      (void*)indptr.ptr(0),
                      crd_domain.empty() ? nullptr : getPtrFromStore<INDEX_TY, 1>(crd),
                      vals.domain().empty() ? nullptr : getPtrFromStore<VAL_TY, 1>(vals),
                      cusparseIndexType<int64_t>(),
                      cusparseIndexType<INDEX_TY>(),
                      index_base,
                      cusparseDataType<VAL_TY>()));
#endif
  return matDescr;
}

// makeCuSparseDenseVec creates a cuSparse dense vector from an input store.
template <typename VAL_TY = double>
cusparseDnVecDescr_t makeCuSparseDenseVec(const legate::Store& vec)
{
  cusparseDnVecDescr_t vecDescr;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecDescr,
                                     vec.domain().get_volume() /* size */,
                                     getPtrFromStore<VAL_TY, 1>(vec),
                                     cusparseDataType<VAL_TY>()));
  return vecDescr;
}

// makeCuSparseDenseMat creates a cuSparse dense vector from an input store.
template <typename VAL_TY = double>
cusparseDnMatDescr_t makeCuSparseDenseMat(const legate::Store& mat)
{
  auto d = mat.domain();

  // Change how we get the pointer based on the privilege of the input store.
  VAL_TY* valsPtr = nullptr;
  size_t ld       = 0;
  if (mat.is_writable() && mat.is_readable()) {
    auto acc = mat.read_write_accessor<VAL_TY, 2>();
    valsPtr  = acc.ptr(d.lo());
    ld       = acc.accessor.strides[0] / sizeof(VAL_TY);
  } else if (mat.is_writable() && !mat.is_readable()) {
    auto acc = mat.write_accessor<VAL_TY, 2>();
    valsPtr  = acc.ptr(d.lo());
    ld       = acc.accessor.strides[0] / sizeof(VAL_TY);
  } else if (!mat.is_writable() && mat.is_readable()) {
    auto acc = mat.read_accessor<VAL_TY, 2>();
    valsPtr  = const_cast<VAL_TY*>(acc.ptr(d.lo()));
    ld       = acc.accessor.strides[0] / sizeof(VAL_TY);
  } else if (mat.is_reducible()) {
    auto acc = mat.reduce_accessor<Legion::SumReduction<VAL_TY>, true /* exclusive */, 2>();
    valsPtr  = acc.ptr(d.lo());
    ld       = acc.accessor.strides[0] / sizeof(VAL_TY);
  } else {
    assert(false);
  }

  cusparseDnMatDescr_t matDescr;
  CHECK_CUSPARSE(cusparseCreateDnMat(&matDescr,
                                     d.hi()[0] - d.lo()[0] + 1, /* rows */
                                     d.hi()[1] - d.lo()[1] + 1, /* columns */
                                     ld,
                                     (void*)valsPtr,
                                     cusparseDataType<VAL_TY>(),
                                     CUSPARSE_ORDER_ROW));
  return matDescr;
}

// cast is a small utility kernel to cast an array of one type into another.
template <typename T1, typename T2>
__global__ void cast(size_t elems, T1* out, const T2* in)
{
  const auto idx = global_tid_1d();
  if (idx >= elems) return;
  out[idx] = T1(in[idx]);
}

// localIndPtrToNnz is a utility kernel to turn a cuSPARSE computed
// indptr array into an nnz array.
template <typename T>
__global__ void localIndptrToNnz(size_t rows, uint64_t* out, T* in)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  out[idx] = in[idx + 1] - in[idx];
}

// localIndptrToPos is a utility kernel to cast an indptr array
// into a legate.sparse pos array.
template <typename T>
__global__ void localIndptrToPos(size_t rows, Legion::Rect<1>* out, T* in)
{
  const auto idx = global_tid_1d();
  if (idx >= rows) return;
  out[idx] = {in[idx], in[idx + 1] - 1};
}

}  // namespace sparse
