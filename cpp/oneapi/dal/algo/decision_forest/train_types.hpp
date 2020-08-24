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

#include "oneapi/dal/algo/decision_forest/common.hpp"

namespace oneapi::dal::decision_forest {

namespace detail {
template <typename Task = task::by_default>
class train_input_impl;

template <typename Task = task::by_default>
class train_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class train_input : public base {
public:
    using pimpl = dal::detail::pimpl<detail::train_input_impl<Task>>;
    train_input(const table& data, const table& labels);

    table get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    table get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

private:
    void set_data_impl(const table& value);
    void set_labels_impl(const table& value);

    pimpl impl_;
};

template <typename Task = task::by_default>
class train_result {
public:
    using pimpl = dal::detail::pimpl<detail::train_result_impl<Task>>;
    train_result();

    model<Task> get_model() const;

    table get_oob_err() const;
    table get_oob_err_per_observation() const;
    table get_var_importance() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    auto& set_oob_err(const table& value) {
        set_oob_err_impl(value);
        return *this;
    }

    auto& set_oob_err_per_observation(const table& value) {
        set_oob_err_per_observation_impl(value);
        return *this;
    }

    auto& set_var_importance(const table& value) {
        set_var_importance_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<Task>&);

    void set_oob_err_impl(const table&);
    void set_oob_err_per_observation_impl(const table&);
    void set_var_importance_impl(const table&);

    pimpl impl_;
};

} // namespace oneapi::dal::decision_forest
