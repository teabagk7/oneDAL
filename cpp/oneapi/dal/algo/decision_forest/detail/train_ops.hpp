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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::decision_forest::detail {

template <typename Context, typename Float, typename Task, typename Method>
struct ONEAPI_DAL_EXPORT train_ops_dispatcher {
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using input_t = train_input<task_t>;
    using result_t = train_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        if (!(input.get_data().has_data())) {
            throw domain_error("Input data should not be empty");
        }
        if (!(input.get_labels().has_data())) {
            throw domain_error("Input labels should not be empty");
        }
        if (input.get_data().get_row_count() != input.get_labels().get_row_count()) {
            throw invalid_argument("Input data row_count should be equal to labels row_count");
        }
        if (!params.get_bootstrap() &&
            (params.get_variable_importance_mode() == variable_importance_mode::mda_raw ||
             params.get_variable_importance_mode() == variable_importance_mode::mda_scaled)) {
            throw invalid_argument(
                "Parameter 'bootstrap' is incompatible with requested variable importance mode");
        }

        if (!params.get_bootstrap() &&
            (check_mask_flag(params.get_error_metric_mode(), error_metric_mode::out_of_bag_error) ||
             check_mask_flag(params.get_error_metric_mode(),
                             error_metric_mode::out_of_bag_error_per_observation))) {
            throw invalid_argument(
                "Parameter 'bootstrap' is incompatible with requested OOB result (no out-of-bag observations)");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            train_ops_dispatcher<Context, float_t, task_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::decision_forest::detail
