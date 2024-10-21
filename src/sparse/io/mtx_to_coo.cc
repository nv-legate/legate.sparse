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

#include "sparse/io/mtx_to_coo.h"

#include <fstream>

namespace sparse {

using namespace legate;

// TODO (rohany): The launching task should tell us what
//  type of data structures to allocate by looking at the
//  matrix market file.
using coord_ty = int64_t;
using val_ty   = double;

/*static*/ void ReadMTXToCOO::cpu_variant(TaskContext& ctx)
{
  // Much of this code was adapted from the matrix-market file IO module
  // within DISTAL.
  assert(ctx.is_single_task());
  // Regardless of how inputs are added, scalar future return values are at the front.
  auto& m_store   = ctx.outputs()[0];
  auto& n_store   = ctx.outputs()[1];
  auto& nnz_store = ctx.outputs()[2];
  auto& rows      = ctx.outputs()[3];
  auto& cols      = ctx.outputs()[4];
  auto& vals      = ctx.outputs()[5];
  auto filename   = ctx.scalars()[0].value<std::string>();
  std::fstream file;
  file.open(filename, std::fstream::in);

  // Parse the header. The header is structured as follows:
  //  %%MatrixMarket type format field symmetry
  std::string line;
  std::getline(file, line);
  std::stringstream lineStream(line);
  std::string head, type, formats, field, symmetry;
  lineStream >> head >> type >> formats >> field >> symmetry;
  assert(head == "%%MatrixMarket" && "Unknown header of MatrixMarket");
  assert(type == "matrix" && "must have type matrix");
  assert(formats == "coordinate" && "must be coordinate");
  enum ValueKind {
    REAL,
    PATTERN,
    INTEGER,
  };
  ValueKind valueKind;
  if (field == "real") {
    valueKind = REAL;
  } else if (field == "pattern") {
    valueKind = PATTERN;
  } else if (field == "integer") {
    valueKind = INTEGER;
  } else {
    assert(false && "unknown field");
  }
  bool symmetric = false;
  if (symmetry == "symmetric") {
    symmetric = true;
  } else if (symmetry == "general") { /* Do nothing. */
  } else {
    assert(false && "unknown symmetry");
  }

  // Skip comments at the top of the file.
  std::string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') { break; }
  } while (std::getline(file, line));

  char* linePtr = (char*)line.data();
  coord_ty m, n;
  size_t lines;
  {
    std::vector<coord_ty> dimensions;
    while (size_t dimension = strtoull(linePtr, &linePtr, 10)) {
      dimensions.push_back(static_cast<coord_ty>(dimension));
    }
    m     = dimensions[0];
    n     = dimensions[1];
    lines = dimensions[2];
  }

  size_t bufSize = lines;
  if (symmetric) { bufSize *= 2; }

  auto row_acc  = rows.create_output_buffer<coord_ty, 1>(bufSize, true /* return_data */);
  auto col_acc  = cols.create_output_buffer<coord_ty, 1>(bufSize, true /* return_data */);
  auto vals_acc = vals.create_output_buffer<val_ty, 1>(bufSize, true /* return_data */);

  size_t idx = 0;
  while (std::getline(file, line)) {
    char* linePtr   = (char*)line.data();
    coord_ty coordX = strtoll(linePtr, &linePtr, 10);
    coord_ty coordY = strtoll(linePtr, &linePtr, 10);
    // MTX coordinates 1 indexed rather than 0 indexed.
    row_acc[idx] = coordX - 1;
    col_acc[idx] = coordY - 1;
    double val;
    if (valueKind == PATTERN) {
      val = 1.0;
    } else if (valueKind == INTEGER) {
      val = strtol(linePtr, &linePtr, 10);
    } else {
      val = strtod(linePtr, &linePtr);
    }
    vals_acc[idx] = val;
    idx++;
    if (symmetric && coordX != coordY) {
      row_acc[idx]  = coordY - 1;
      col_acc[idx]  = coordX - 1;
      vals_acc[idx] = val;
      idx++;
    }
  }

  file.close();
  m_store.write_accessor<int64_t, 1>()[0]    = int64_t(m);
  n_store.write_accessor<int64_t, 1>()[0]    = int64_t(n);
  nnz_store.write_accessor<uint64_t, 1>()[0] = uint64_t(idx);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { ReadMTXToCOO::register_variants(); }
}  // namespace

}  // namespace sparse
