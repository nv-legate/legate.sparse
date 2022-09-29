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

#include "sparse.h"
#include "sparse_c.h"
#include "legate.h"

namespace sparse {

class EnumerateIndependentSets : public SparseTask<EnumerateIndependentSets> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_ENUMERATE_INDEP_SETS;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  template<int N, typename T>
  static void omp_variant_impl(legate::TaskContext& ctx);
#endif
};

class CreateHamiltonians : public SparseTask<CreateHamiltonians> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_CREATE_HAMILTONIANS;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  template<int N, typename T>
  static void omp_variant_impl(legate::TaskContext& ctx);
#endif
};

class SetsToSizes : public SparseTask<SetsToSizes> {
public:
  static const int TASK_ID = LEGATE_QUANTUM_SETS_TO_SIZES;
  static void cpu_variant(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& ctx);
#endif
private:
  template<int N, typename T>
  static void cpu_variant_impl(legate::TaskContext& ctx);
#ifdef LEGATE_USE_OPENMP
  template<int N, typename T>
  static void omp_variant_impl(legate::TaskContext& ctx);
#endif
};

// IntSet is a small-integer set stored using bit masks.
template<int N, typename T>
struct IntSet {
public:
  IntSet() {
    for (int i = 0; i < N; i++) {
      bits[i] = T(0);
    }
  }

  IntSet<N, T> set_index(int idx) const {
    IntSet<N, T> result = *this;
    auto arr_idx = idx / (sizeof(T) * 8);
    auto in_arr_idx = idx % (sizeof(T) * 8);
    T bit = T(1) << in_arr_idx;
    result.bits[arr_idx] |= bit;
    return result;
  }

  IntSet<N, T> unset_index(int idx) const {
    IntSet<N, T> result = *this;
    auto arr_idx = idx / (sizeof(T) * 8);
    auto in_arr_idx = idx % (sizeof(T) * 8);
    T bit = T(1) << in_arr_idx;
    result.bits[arr_idx] &= ~bit;
    return result;
  }

  bool is_index_set(int idx) const {
    auto arr_idx = idx / (sizeof(T) * 8);
    auto in_arr_idx = idx % (sizeof(T) * 8);
    T bit = T(1) << in_arr_idx;
    return (this->bits[arr_idx] & bit) > T(0);
  }

  int get_set_bits() const {
    int count = 0;
    for (int i = 0; i < (N * sizeof(T) * 8); i++) {
      count += int(this->is_index_set(i));
    }
    return count;
  }

  friend bool operator< (const IntSet<N, T>& a, const IntSet<N, T>& b) {
    return a.bits < b.bits;
  }

  friend bool operator== (const IntSet<N, T>& a, const IntSet<N, T>& b) {
    return a.bits == b.bits;
  }

  std::array<T, N> bits;
};


// bit_ty defines what size the individual words within the IntSet should
// be. The larger the bit_ty type, the fewer template instantiations we
// need, but also the larger the jump in memory usage is when we increase
// the number of words needed in the bitset. I'm starting with uint8_t for
// now since binary size isn't really a problem for us.
using bit_ty = uint8_t;

// A macro to perform the dispatch to the right template instantiation of IntSet.
#define INTSET_DISPATCH(x, n) \
  switch ((n + (sizeof(bit_ty) * 8) - 1) / (sizeof(bit_ty) * 8)) { \
    case 1: \
      x<1, bit_ty>(ctx); \
      break; \
    case 2: \
      x<2, bit_ty>(ctx); \
      break; \
    case 3: \
      x<3, bit_ty>(ctx); \
      break; \
    case 4: \
      x<4, bit_ty>(ctx); \
      break; \
    case 5: \
      x<5, bit_ty>(ctx); \
      break; \
    case 6: \
      x<6, bit_ty>(ctx); \
      break; \
    case 7: \
      x<7, bit_ty>(ctx); \
      break; \
    case 8: \
      x<8, bit_ty>(ctx); \
      break; \
    case 9: \
      x<9, bit_ty>(ctx); \
      break; \
    case 10: \
      x<10, bit_ty>(ctx); \
      break; \
    case 11: \
      x<11, bit_ty>(ctx); \
      break; \
    case 12: \
      x<12, bit_ty>(ctx); \
      break; \
    default: \
      assert(false); \
  }

}
