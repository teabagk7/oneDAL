/*******************************************************************************
* Copyright 2020 Intel Corporation
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
*******************************************************************************/

#pragma once

#include <type_traits>

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

template <typename T>
constexpr data_type make_data_type_impl();

template <typename T>
constexpr data_type make_data_type() {
    return make_data_type_impl<std::decay_t<T>>();
}

constexpr std::int64_t get_data_type_size(data_type t) {
    if (t == data_type::float32) {
        return sizeof(float);
    }
    else if (t == data_type::float64) {
        return sizeof(double);
    }
    else if (t == data_type::int32) {
        return sizeof(int32_t);
    }
    else if (t == data_type::int64) {
        return sizeof(int64_t);
    }
    else if (t == data_type::uint32) {
        return sizeof(uint32_t);
    }
    else if (t == data_type::uint64) {
        return sizeof(uint64_t);
    }
    return 0;
}

constexpr bool is_floating_point(data_type t) {
    if (t == data_type::bfloat16 || t == data_type::float32 || t == data_type::float64) {
        return true;
    }
    else {
        return false;
    }
}

template <typename T>
constexpr bool is_floating_point() {
    return is_floating_point(make_data_type<T>());
}

template <typename T>
constexpr data_type make_data_type_impl() {
    if constexpr (std::is_same_v<std::int32_t, T>) {
        return data_type::int32;
    }
    else if constexpr (std::is_same_v<std::int64_t, T>) {
        return data_type::int64;
    }
    else if constexpr (std::is_same_v<std::uint32_t, T>) {
        return data_type::uint32;
    }
    else if constexpr (std::is_same_v<std::uint64_t, T>) {
        return data_type::uint64;
    }
    else if constexpr (std::is_same_v<float, T>) {
        return data_type::float32;
    }
    else if constexpr (std::is_same_v<double, T>) {
        return data_type::float64;
    }

    static_assert(
        is_one_of<T, std::int32_t, std::int64_t, std::uint32_t, std::uint64_t, float, double>::
            value,
        "unsupported data type");
    return data_type::float32; // shall never come here
}

} // namespace oneapi::dal::detail
