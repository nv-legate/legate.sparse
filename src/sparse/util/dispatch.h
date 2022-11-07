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

#include "core/utilities/typedefs.h"

namespace sparse {

template <legate::LegateTypeCode INDEX_TY_CODE, typename Functor, typename... Fnargs>
constexpr decltype(auto) value_type_dispatch_from_index(legate::LegateTypeCode value_type,
                                                        Functor f,
                                                        Fnargs&&... args)
{
  // Dispatch on the supported value types, conditioned on the index types.
  switch (value_type) {
    case legate::LegateTypeCode::FLOAT_LT: {
      return f.template operator()<INDEX_TY_CODE, legate::LegateTypeCode::FLOAT_LT>(
        std::forward<Fnargs>(args)...);
    }
    case legate::LegateTypeCode::DOUBLE_LT: {
      return f.template operator()<INDEX_TY_CODE, legate::LegateTypeCode::DOUBLE_LT>(
        std::forward<Fnargs>(args)...);
    }
    case legate::LegateTypeCode::COMPLEX64_LT: {
      return f.template operator()<INDEX_TY_CODE, legate::LegateTypeCode::COMPLEX64_LT>(
        std::forward<Fnargs>(args)...);
    }
    case legate::LegateTypeCode::COMPLEX128_LT: {
      return f.template operator()<INDEX_TY_CODE, legate::LegateTypeCode::COMPLEX128_LT>(
        std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<INDEX_TY_CODE, legate::LegateTypeCode::FLOAT_LT>(
    std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) index_type_value_type_dispatch(legate::LegateTypeCode index_type,
                                                        legate::LegateTypeCode value_type,
                                                        Functor f,
                                                        Fnargs&&... args)
{
  // First dispatch onto the index type.
  switch (index_type) {
    case legate::LegateTypeCode::INT32_LT: {
      return value_type_dispatch_from_index<legate::LegateTypeCode::INT32_LT, Functor, Fnargs...>(
        value_type, f, args...);
    }
    case legate::LegateTypeCode::INT64_LT: {
      return value_type_dispatch_from_index<legate::LegateTypeCode::INT64_LT, Functor, Fnargs...>(
        value_type, f, args...);
    }
    default: break;
  }
  assert(false);
  return value_type_dispatch_from_index<legate::LegateTypeCode::INT32_LT, Functor, Fnargs...>(
    value_type, f, args...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) index_type_dispatch(legate::LegateTypeCode index_type,
                                             Functor f,
                                             Fnargs&&... args)
{
  // First dispatch onto the index type.
  switch (index_type) {
    case legate::LegateTypeCode::INT32_LT: {
      return f.template operator()<legate::LegateTypeCode::INT32_LT>(std::forward<Fnargs>(args)...);
    }
    case legate::LegateTypeCode::INT64_LT: {
      return f.template operator()<legate::LegateTypeCode::INT64_LT>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<legate::LegateTypeCode::INT32_LT>(std::forward<Fnargs>(args)...);
}

}  // namespace sparse