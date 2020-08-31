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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/detail/model_impl.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::decision_forest {

inline void check_domain_cond(bool value, const char* description) {
    if (!(value))
        throw dal::domain_error(description);
}

template <>
class detail::descriptor_impl<task::classification> : public base {
public:
    double observations_per_tree_fraction = 1.0;
    double impurity_threshold = 0.0;
    double min_weight_fraction_in_leaf_node = 0.0;
    double min_impurity_decrease_in_split_node = 0.0;

    std::int64_t class_count = 2;
    std::int64_t tree_count = 100;
    std::int64_t features_per_node = 0;
    std::int64_t max_tree_depth = 0;
    std::int64_t min_observations_in_leaf_node = 1;
    std::int64_t min_observations_in_split_node = 2;
    std::int64_t max_leaf_nodes = 0;

    error_metric_mode error_metric_mode_value = error_metric_mode::none;
    infer_mode infer_mode_value = infer_mode::class_labels;

    bool memory_saving_mode = false;
    bool bootstrap = true;

    variable_importance_mode variable_importance_mode_value = variable_importance_mode::none;
    voting_mode voting_mode_value = voting_mode::weighted;
};

template <>
class detail::descriptor_impl<task::regression> : public base {
public:
    double observations_per_tree_fraction = 1.0;
    double impurity_threshold = 0.0;
    double min_weight_fraction_in_leaf_node = 0.0;
    double min_impurity_decrease_in_split_node = 0.0;

    std::int64_t class_count = 0;
    std::int64_t tree_count = 100;
    std::int64_t features_per_node = 0;
    std::int64_t max_tree_depth = 0;
    std::int64_t min_observations_in_leaf_node = 5;
    std::int64_t min_observations_in_split_node = 2;
    std::int64_t max_leaf_nodes = 0;

    error_metric_mode error_metric_mode_value = error_metric_mode::none;
    infer_mode infer_mode_value = infer_mode::class_labels;

    bool memory_saving_mode = false;
    bool bootstrap = true;

    // engine field

    variable_importance_mode variable_importance_mode_value = variable_importance_mode::none;
    voting_mode voting_mode_value = voting_mode::weighted;
};

using detail::descriptor_impl;
using detail::model_impl;

/* descriptor_base implementation */

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

/*getters implementation*/
template <typename Task>
double descriptor_base<Task>::get_observations_per_tree_fraction() const {
    return impl_->observations_per_tree_fraction;
}
template <typename Task>
double descriptor_base<Task>::get_impurity_threshold() const {
    return impl_->impurity_threshold;
}
template <typename Task>
double descriptor_base<Task>::get_min_weight_fraction_in_leaf_node() const {
    return impl_->min_weight_fraction_in_leaf_node;
}
template <typename Task>
double descriptor_base<Task>::get_min_impurity_decrease_in_split_node() const {
    return impl_->min_impurity_decrease_in_split_node;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_tree_count() const {
    return impl_->tree_count;
}
template <typename Task>
std::int64_t descriptor_base<Task>::get_features_per_node() const {
    return impl_->features_per_node;
}
template <typename Task>
std::int64_t descriptor_base<Task>::get_max_tree_depth() const {
    return impl_->max_tree_depth;
}
template <typename Task>
std::int64_t descriptor_base<Task>::get_min_observations_in_leaf_node() const {
    return impl_->min_observations_in_leaf_node;
}
template <typename Task>
std::int64_t descriptor_base<Task>::get_min_observations_in_split_node() const {
    return impl_->min_observations_in_split_node;
}
template <typename Task>
std::int64_t descriptor_base<Task>::get_max_leaf_nodes() const {
    return impl_->max_leaf_nodes;
}

template <typename Task>
error_metric_mode descriptor_base<Task>::get_error_metric_mode() const {
    return impl_->error_metric_mode_value;
}

template <typename Task>
bool descriptor_base<Task>::get_memory_saving_mode() const {
    return impl_->memory_saving_mode;
}
template <typename Task>
bool descriptor_base<Task>::get_bootstrap() const {
    return impl_->bootstrap;
}

template <typename Task>
variable_importance_mode descriptor_base<Task>::get_variable_importance_mode() const {
    return impl_->variable_importance_mode_value;
}

template <typename Task>
infer_mode descriptor_base<Task>::get_infer_mode_impl() const {
    return impl_->infer_mode_value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_class_count_impl() const {
    return impl_->class_count;
}
template <typename Task>
voting_mode descriptor_base<Task>::get_voting_mode_impl() const {
    return impl_->voting_mode_value;
}
/*setters implementation*/

template <typename Task>
void descriptor_base<Task>::set_observations_per_tree_fraction_impl(double value) {
    check_domain_cond((value > 0.0 && value <= 1.0),
                      "observations_per_tree_fraction schould be > 0.0 and <= 1.0");
    impl_->observations_per_tree_fraction = value;
}
template <typename Task>
void descriptor_base<Task>::set_impurity_threshold_impl(double value) {
    check_domain_cond((value >= 0.0), "impurity_threshold schould be >= 0.0");
    impl_->impurity_threshold = value;
}
template <typename Task>
void descriptor_base<Task>::set_min_weight_fraction_in_leaf_node_impl(double value) {
    check_domain_cond((value >= 0.0 && value <= 0.5),
                      "min_weight_fraction_in_leaf_node schould be >= 0.0 and <= 0.5");
    impl_->min_weight_fraction_in_leaf_node = value;
}
template <typename Task>
void descriptor_base<Task>::set_min_impurity_decrease_in_split_node_impl(double value) {
    check_domain_cond((value >= 0.0), "min_impurity_decrease_in_split_node schould be >= 0.0");
    impl_->min_impurity_decrease_in_split_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_tree_count_impl(std::int64_t value) {
    check_domain_cond((value > 0), "tree_count schould be > 0");
    impl_->tree_count = value;
}
template <typename Task>
void descriptor_base<Task>::set_features_per_node_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "features_per_node schould be >= 0");
    impl_->features_per_node = value;
}
template <typename Task>
void descriptor_base<Task>::set_max_tree_depth_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "max_tree_depth schould be >= 0");
    impl_->max_tree_depth = value;
}
template <typename Task>
void descriptor_base<Task>::set_min_observations_in_leaf_node_impl(std::int64_t value) {
    check_domain_cond((value > 0), "min_observations_in_leaf_node schould be > 0");
    impl_->min_observations_in_leaf_node = value;
}
template <typename Task>
void descriptor_base<Task>::set_min_observations_in_split_node_impl(std::int64_t value) {
    check_domain_cond((value > 0), "min_observations_in_split_node schould be > 0");
    impl_->min_observations_in_split_node = value;
}
template <typename Task>
void descriptor_base<Task>::set_max_leaf_nodes_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "max_leaf_nodes schould be >= 0");
    impl_->max_leaf_nodes = value;
}

template <typename Task>
void descriptor_base<Task>::set_error_metric_mode_impl(error_metric_mode value) {
    impl_->error_metric_mode_value = value;
}
template <typename Task>
void descriptor_base<Task>::set_infer_mode_impl(infer_mode value) {
    impl_->infer_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_memory_saving_mode_impl(bool value) {
    impl_->memory_saving_mode = value;
}
template <typename Task>
void descriptor_base<Task>::set_bootstrap_impl(bool value) {
    impl_->bootstrap = value;
}

template <typename Task>
void descriptor_base<Task>::set_variable_importance_mode_impl(variable_importance_mode value) {
    impl_->variable_importance_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_class_count_impl(std::int64_t value) {
    check_domain_cond((value > 0), "class_count schould be > 0");
    impl_->class_count = value;
}
template <typename Task>
void descriptor_base<Task>::set_voting_mode_impl(voting_mode value) {
    impl_->voting_mode_value = value;
}

template class ONEAPI_DAL_EXPORT descriptor_base<task::classification>;
template class ONEAPI_DAL_EXPORT descriptor_base<task::regression>;

/* model implementation */
template <typename Task>
model<Task>::model() : impl_(new detail::model_impl<Task>{}) {}
template <typename Task>
model<Task>::model(const model::pimpl& impl) : impl_(impl) {}

template <typename Task>
std::int64_t model<Task>::get_tree_count() const {
    return impl_->get_tree_count();
}
template <typename Task>
std::int64_t model<Task>::get_class_count_impl() const {
    return impl_->get_class_count();
}
template <typename Task>
void model<Task>::clear() {
    impl_->clear();
}

template class ONEAPI_DAL_EXPORT model<task::classification>;
template class ONEAPI_DAL_EXPORT model<task::regression>;
} // namespace oneapi::dal::decision_forest
