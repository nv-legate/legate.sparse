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

#include "sparse/util/upcast_future.h"

namespace sparse {

using namespace legate;

template <typename T>
void upcast_impl(legate::TaskContext& ctx)
{
  auto& in_fut = ctx.inputs()[0];
  const T* src;
  T* dst;
  switch (in_fut.dim()) {
    case 0: {
      // Futures can be 0-dimensional. legate doesn't appear to complain
      // if we make a 1-D accessor of a 0-D "store".
      dst = ctx.outputs()[0].write_accessor<T, 1, false>().ptr(0);
      src = ctx.inputs()[0].read_accessor<T, 1, false>().ptr(0);
      break;
    }
    case 1: {
      dst = ctx.outputs()[0].write_accessor<T, 1, false>().ptr(0);
      src = ctx.inputs()[0].read_accessor<T, 1, false>().ptr(0);
      break;
    }
    case 2: {
      dst = ctx.outputs()[0].write_accessor<T, 2, false>().ptr({0, 0});
      src = ctx.inputs()[0].read_accessor<T, 2, false>().ptr({0, 0});
      break;
    }
    case 3: {
      dst = ctx.outputs()[0].write_accessor<T, 3, false>().ptr({0, 0, 0});
      src = ctx.inputs()[0].read_accessor<T, 3, false>().ptr({0, 0, 0});
      break;
    }
  }
  memcpy(dst, src, sizeof(T));
}

/*static*/ void UpcastFutureToRegion::cpu_variant(TaskContext& ctx)
{
  assert(ctx.is_single_task());
  auto future_size = ctx.scalars()[0].value<size_t>();
  switch (future_size) {
    case 1: {
      upcast_impl<uint8_t>(ctx);
      break;
    }
    case 2: {
      upcast_impl<uint16_t>(ctx);
      break;
    }
    case 4: {
      upcast_impl<uint32_t>(ctx);
      break;
    }
    case 8: {
      upcast_impl<uint64_t>(ctx);
      break;
    }
    default: {
      assert(false);
    }
  }
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  UpcastFutureToRegion::register_variants();
}
}  // namespace

}  // namespace sparse
