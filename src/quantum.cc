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

#include "tasks.h"
#include "sparse.h"
#include "quantum.h"

using namespace Legion;

namespace sparse {

typedef uint64_t set_ty;

inline set_ty set_index(set_ty set, int64_t idx) {
  uint64_t bit = 1;
  return set | (bit << idx);
}

inline set_ty unset_index(set_ty set, int64_t idx) {
  uint64_t bit = 1;
  return set & ~(bit << idx);
}

inline bool is_index_set(set_ty set, int64_t idx) {
  uint64_t bit = 1;
  return (set & (bit << idx)) > 0;
}

inline int64_t get_set_bits(set_ty set) {
  int count = 0;
  for (int i = 0; i < 64; i++) {
    count += int(is_index_set(set, i));
  }
  return count;
}

void EnumerateIndependentSets::cpu_variant(legate::TaskContext& ctx) {
  auto& graph = ctx.inputs()[0];
  auto graph_acc = graph.read_accessor<int64_t, 2>();
  int64_t nodes = graph.domain().hi()[0] - graph.domain().lo()[0] + 1;

  // Set up input data structures, similar to enumerate_independent_sets in
  // graph_tools.py.
  std::map<int64_t, int64_t> index;
  std::map<int64_t, std::vector<int64_t>> nbrs;
  for (int64_t node = 0; node < nodes; node++) {
    auto size = index.size();
    index[node] = size;
    std::vector<int64_t> neighbors;
    for (int64_t other_node = 0; other_node < nodes; other_node++) {
      if (graph_acc[{node, other_node}] && index.find(other_node) == index.end()) {
        neighbors.push_back(other_node);
      }
    }
    nbrs[node] = neighbors;
  }

  auto& output_sets = ctx.outputs()[0];
  auto& output_nbrs = ctx.outputs()[1];

  int32_t k = ctx.scalars()[0].value<int32_t>();
  if (k == 1) {
    assert(ctx.is_single_task());
    // If k == 1, then the resulting output sets are just the nodes.
    auto output_sets_acc = output_sets.create_output_buffer<set_ty, 1>(nodes, true /* return_buffer */);
    auto output_nbrs_acc = output_nbrs.create_output_buffer<set_ty, 1>(nodes, true /* return_buffer */);
    for (int64_t node = 0; node < nodes; node++) {
      output_sets_acc[node] = set_index(0, node);
      set_ty node_nbrs = 0;
      for (auto nbr : nbrs[node]) {
        node_nbrs = set_index(node_nbrs, nbr);
      }
      output_nbrs_acc[node] = node_nbrs;
    }
  } else {
    auto& prev_sets = ctx.inputs()[1];
    auto& prev_nbrs = ctx.inputs()[2];
    auto prev_sets_acc = prev_sets.read_accessor<set_ty, 1>();
    auto prev_nbrs_acc = prev_nbrs.read_accessor<set_ty, 1>();

    int64_t count = 0;
    // Calculate the total number of outputs we'll generate.
    for (PointInDomainIterator<1> itr(prev_sets.domain()); itr(); itr++) {
      // We will generate an entry for each of the neighbors in the sets
      // that exist at the current iteration.
      count += get_set_bits(prev_nbrs_acc[*itr]);
    }
    auto output_sets_acc = output_sets.create_output_buffer<set_ty, 1>(count, true /* return_buffer */);
    auto output_nbrs_acc = output_nbrs.create_output_buffer<set_ty, 1>(count, true /* return_buffer */);
    int64_t idx = 0;
    for (PointInDomainIterator<1> itr(prev_sets.domain()); itr(); itr++) {
      auto prev_set = prev_sets_acc[*itr];
      auto prev_nbrs = prev_nbrs_acc[*itr];

      // TODO (rohany): See if there is a better primitive to loop over
      //  all of the set bits.
      for (int u = 0; u < nodes; u++) {
        if (is_index_set(prev_nbrs, u)) {
          set_ty new_nbrs = 0;
          output_sets_acc[idx] = set_index(prev_set, u);
          // The new neighbors is the intersection of all remaining
          // neighbors and i's neighbors.
          for (int v = u + 1; v < nodes; v++) {
            if (is_index_set(prev_nbrs, v) && graph_acc[{u, v}]) {
              new_nbrs = set_index(new_nbrs, v);
            }
          }
          output_nbrs_acc[idx] = new_nbrs;
          idx++;
        }
      }
    }
  }
}

void CreateHamiltonians::cpu_variant(legate::TaskContext& ctx) {
  int32_t k = ctx.scalars()[0].value<int32_t>();
  int32_t nodes = ctx.scalars()[1].value<int32_t>();
  uint64_t set_idx_offset = ctx.scalars()[2].value<uint64_t>();
  auto& rows = ctx.outputs()[0];
  auto& cols = ctx.outputs()[1];
  auto& sets = ctx.inputs()[0];
  auto sets_acc = sets.read_accessor<set_ty, 1>();

  if (k == 1) {
    // If k == 1, then we don't have any predecessor states to look
    // at and can just add on the terminal state to each entry.
    auto volume = sets.domain().get_volume();
    auto rows_acc = rows.create_output_buffer<coord_ty, 1>(2 * volume, true /* return_buffer */);
    auto cols_acc = cols.create_output_buffer<coord_ty, 1>(2 * volume, true /* return_buffer */);
    for (coord_ty i = 0; i < volume; i++) {
      auto set_idx = set_idx_offset + sets.domain().lo()[0] + i;
      // The predecessors of size 1 index sets are the null state 0.
      rows_acc[2 * i] = set_idx;
      cols_acc[2 * i] = 0;
      rows_acc[2 * i + 1] = 0;
      cols_acc[2 * i + 1] = set_idx;
    }
  } else {
    auto& preds = ctx.inputs()[1];
    auto preds_acc = preds.read_accessor<set_ty, 1>();
    uint64_t preds_idx_offset = ctx.scalars()[3].value<uint64_t>();

    // Put all the predecessors into a map with their index so
    // that we can look up into it quickly.
    // TODO (rohany): We'll have to go with sort + binary searching
    //  for OpenMP and GPU processors.
    std::map<set_ty, uint64_t> index;
    for (PointInDomainIterator<1> itr(preds.domain()); itr(); itr++) {
      index[preds_acc[*itr]] = preds_idx_offset + (*itr);
    }

    // Find out the number coordinates each set will output.
    uint64_t count = 0;
    for (PointInDomainIterator<1> itr(sets.domain()); itr(); itr++) {
      auto set = sets_acc[*itr];
      for (int32_t node = 0; node < nodes; node++) {
        if (is_index_set(set, node)) {
          // Try to find the set without this node in the predecessors.
          auto removed = unset_index(set, node);
          if (index.find(removed) != index.end()) {
            count++;
          }
        }
      }
    }
    // Each coordinate actually counts for 2, since this is a bi-directional graph.
    auto rows_acc = rows.create_output_buffer<coord_ty, 1>(2 * count, true /* return_buffer */);
    auto cols_acc = cols.create_output_buffer<coord_ty, 1>(2 * count, true /* return_buffer */);
    // Calculate the coordinates.
    uint64_t slot = 0;
    for (PointInDomainIterator<1> itr(sets.domain()); itr(); itr++) {
      auto set = sets_acc[*itr];
      for (int32_t node = 0; node < nodes; node++) {
        if (is_index_set(set, node)) {
          // Try to find the set without this node in the predecessors.
          auto removed = unset_index(set, node);
          auto query = index.find(removed);
          if (query != index.end()) {
            auto pred_idx = query->second;
            auto set_idx = set_idx_offset + (*itr);
            rows_acc[slot] = set_idx;
            cols_acc[slot] = pred_idx;
            rows_acc[slot + 1] = pred_idx;
            cols_acc[slot + 1] = set_idx;
            slot += 2;
          }
        }
      }
    }
  }
}

void ExpandSet::cpu_variant(legate::TaskContext &ctx) {
  auto& input = ctx.inputs()[0];
  auto input_acc = input.read_accessor<set_ty, 1>();
  auto dom = input.domain();
  // TODO (rohany): For now we'll just print it...
  for (PointInDomainIterator<1> itr(dom); itr(); itr++) {
    set_ty val = input_acc[*itr];
    // TODO (rohany): This will have to change when the number of nodes is >= 64.
    std::stringstream output;
    output << "[";
    for (int i = 0; i < 64; i++) {
      if (is_index_set(val, i)) {
        output << i << ", ";
      }
    }
    output << "]";
    std::cout << (*itr) << " -> " << output.str() << std::endl;
  }
}

void SetsToSizes::cpu_variant(legate::TaskContext &ctx) {
  auto& input = ctx.inputs()[0];
  auto& output = ctx.outputs()[0];
  auto input_acc = input.read_accessor<set_ty, 1>();
  auto output_acc = output.write_accessor<uint64_t, 1>();
  for (PointInDomainIterator<1> itr(input.domain()); itr(); itr++) {
    output_acc[*itr] = get_set_bits(input_acc[*itr]);
  }
}

}

namespace { // anonymous
static void __attribute__((constructor)) register_tasks(void) {
  sparse::EnumerateIndependentSets::register_variants();
  sparse::ExpandSet::register_variants();
  sparse::SetsToSizes::register_variants();
  sparse::CreateHamiltonians::register_variants();
}
}
