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

template <legate::Type::Code INDEX_TY_CODE, typename Functor, typename... Fnargs>
constexpr decltype(auto) value_type_dispatch_from_index(legate::Type::Code value_type,
                                                        Functor f,
                                                        Fnargs&&... args)
{
  // Dispatch on the supported value types, conditioned on the index types.
  switch (value_type) {
    case legate::Type::Code::FLOAT32: {
      return f.template operator()<INDEX_TY_CODE, legate::Type::Code::FLOAT32>(
        std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT64: {
      return f.template operator()<INDEX_TY_CODE, legate::Type::Code::FLOAT64>(
        std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::COMPLEX64: {
      return f.template operator()<INDEX_TY_CODE, legate::Type::Code::COMPLEX64>(
        std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::COMPLEX128: {
      return f.template operator()<INDEX_TY_CODE, legate::Type::Code::COMPLEX128>(
        std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<INDEX_TY_CODE, legate::Type::Code::FLOAT32>(
    std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) index_type_value_type_dispatch(legate::Type::Code index_type,
                                                        legate::Type::Code value_type,
                                                        Functor f,
                                                        Fnargs&&... args)
{
  // First dispatch onto the index type.
  switch (index_type) {
    case legate::Type::Code::INT32: {
      return value_type_dispatch_from_index<legate::Type::Code::INT32, Functor, Fnargs...>(
        value_type, f, args...);
    }
    case legate::Type::Code::INT64: {
      return value_type_dispatch_from_index<legate::Type::Code::INT64, Functor, Fnargs...>(
        value_type, f, args...);
    }
    default: break;
  }
  assert(false);
  return value_type_dispatch_from_index<legate::Type::Code::INT32, Functor, Fnargs...>(
    value_type, f, args...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) index_type_dispatch(legate::Type::Code index_type,
                                             Functor f,
                                             Fnargs&&... args)
{
  switch (index_type) {
    case legate::Type::Code::INT32: {
      return f.template operator()<legate::Type::Code::INT32>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::INT64: {
      return f.template operator()<legate::Type::Code::INT64>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<legate::Type::Code::INT32>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) value_type_dispatch(legate::Type::Code value_type,
                                             Functor f,
                                             Fnargs&&... args)
{
  switch (value_type) {
    case legate::Type::Code::FLOAT32: {
      return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT64: {
      return f.template operator()<legate::Type::Code::FLOAT64>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::COMPLEX64: {
      return f.template operator()<legate::Type::Code::COMPLEX64>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::COMPLEX128: {
      return f.template operator()<legate::Type::Code::COMPLEX128>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) value_type_dispatch_no_complex(legate::Type::Code value_type,
                                                        Functor f,
                                                        Fnargs&&... args)
{
  switch (value_type) {
    case legate::Type::Code::FLOAT32: {
      return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT64: {
      return f.template operator()<legate::Type::Code::FLOAT64>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
}

}  // namespace sparse
