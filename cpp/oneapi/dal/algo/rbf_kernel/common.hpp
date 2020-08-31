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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::rbf_kernel {

namespace detail {
struct tag {};
class descriptor_impl;
class model_impl;
} // namespace detail

namespace method {
struct dense {};
struct csr {};
using by_default = dense;
} // namespace method

class ONEAPI_DAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;

    descriptor_base();

    double get_sigma() const;

protected:
    void set_sigma_impl(double);

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float = descriptor_base::float_t, typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using float_t = Float;
    using method_t = Method;

    auto& set_sigma(double value) {
        set_sigma_impl(value);
        return *this;
    }
};

} // namespace oneapi::dal::rbf_kernel
