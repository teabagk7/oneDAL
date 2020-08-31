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

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::decision_forest::detail {

template <typename Context, typename Float, typename Task, typename Method>
struct ONEAPI_DAL_EXPORT infer_ops_dispatcher {
    infer_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const infer_input<Task>&) const;
};

template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using input_t = infer_input<task_t>;
    using result_t = infer_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        if (!(input.get_data().has_data())) {
            throw domain_error("Input data should not be empty");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            infer_ops_dispatcher<Context, float_t, task_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::decision_forest::detail
