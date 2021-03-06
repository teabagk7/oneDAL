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

#include "oneapi/dal/algo/pca/train_types.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::pca::detail {

template <typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT train_ops_dispatcher {
    train_result operator()(const Context&, const descriptor_base&, const train_input&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = train_input;
    using result_t = train_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        if (!(input.get_data().has_data())) {
            throw domain_error("Input data should not be empty");
        }
        if (input.get_data().get_column_count() < params.get_component_count()) {
            throw invalid_argument(
                "Input data column_count should be >= descriptor component_count");
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        if (result.get_explained_variance().has_data()) {
            if (result.get_explained_variance().get_row_count() != 1) {
                throw internal_error("Result explained variance row_count should be equal to 1");
            }
            if (result.get_explained_variance().get_column_count() !=
                params.get_component_count()) {
                throw internal_error(
                    "Result explained variance column_count should be equal to descriptor component_count");
            }

            auto arr_examplained_variance =
                row_accessor<const float_t>{ result.get_explained_variance() }.pull();
            for (std::int64_t i = 0; i < result.get_explained_variance().get_column_count(); ++i) {
                if (arr_examplained_variance[i] < 0) {
                    throw internal_error("Result explained variance should be >= 0");
                }
            }
        }
        if (!(result.get_eigenvalues().has_data())) {
            throw internal_error("Result eigenvalues should not be empty");
        }
        if (result.get_eigenvalues().get_row_count() != 1) {
            throw internal_error("Result eigenvalues row_count should be equal to 1");
        }
        if (result.get_eigenvalues().get_column_count() != params.get_component_count()) {
            throw internal_error(
                "Result eigenvalues row_count should be equal to descriptor compunent_count");
        }

        if (!(result.get_eigenvectors().has_data())) {
            throw internal_error("Result eigenvectors should not be empty");
        }
        if (result.get_eigenvectors().get_row_count() != params.get_component_count()) {
            throw internal_error(
                "Result eigenvectors row_count should be equal to descriptor compunent_count");
        }
        if (result.get_eigenvectors().get_column_count() != params.get_component_count()) {
            throw internal_error(
                "Result eigenvectors row_count should be equal to descriptor compunent_count");
        }
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const train_input& input) const {
        check_preconditions(desc, input);
        const auto result = train_ops_dispatcher<Context, float_t, method_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace oneapi::dal::pca::detail
