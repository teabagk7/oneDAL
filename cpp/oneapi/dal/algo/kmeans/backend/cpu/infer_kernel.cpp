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

#include <daal/include/algorithms/kmeans/kmeans_types.h>
#include <daal/src/algorithms/kmeans/kmeans_lloyd_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans::backend {

using std::int64_t;
using dal::backend::context_cpu;

namespace daal_kmeans = daal::algorithms::kmeans;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_lloyd_dense_kernel_t =
    daal_kmeans::internal::KMeansBatchKernel<daal_kmeans::lloydDense, Float, Cpu>;

template <typename Float>
static infer_result call_daal_kernel(const context_cpu& ctx,
                                     const descriptor_base& desc,
                                     const model& trained_model,
                                     const table& data) {
    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    const int64_t cluster_count = desc.get_cluster_count();
    const int64_t max_iteration_count = 0;

    daal_kmeans::Parameter par(cluster_count, max_iteration_count);
    par.resultsToEvaluate = daal_kmeans::computeAssignments;

    auto arr_data = row_accessor<const Float>{ data }.pull();
    auto arr_initial_centroids = row_accessor<const Float>{ trained_model.get_centroids() }.pull();

    array<int> arr_labels = array<int>::empty(row_count);
    array<Float> arr_objective_function_value = array<Float>::empty(1);
    array<int> arr_iteration_count = array<int>::empty(1);

    const auto daal_data = interop::convert_to_daal_homogen_table(arr_data,
                                                                  data.get_row_count(),
                                                                  data.get_column_count());
    const auto daal_initial_centroids =
        interop::convert_to_daal_homogen_table(arr_initial_centroids, cluster_count, column_count);
    const auto daal_labels = interop::convert_to_daal_homogen_table(arr_labels, row_count, 1);
    const auto daal_objective_function_value =
        interop::convert_to_daal_homogen_table(arr_objective_function_value, 1, 1);
    const auto daal_iteration_count =
        interop::convert_to_daal_homogen_table(arr_iteration_count, 1, 1);

    daal::data_management::NumericTable* input[2] = { daal_data.get(),
                                                      daal_initial_centroids.get() };

    daal::data_management::NumericTable* output[4] = { nullptr,
                                                       daal_labels.get(),
                                                       daal_objective_function_value.get(),
                                                       daal_iteration_count.get() };

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_kmeans_lloyd_dense_kernel_t>(ctx,
                                                                           input,
                                                                           output,
                                                                           &par));

    return infer_result()
        .set_labels(dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build())
        .set_objective_function_value(static_cast<double>(arr_objective_function_value[0]));
}

template <typename Float>
static infer_result infer(const context_cpu& ctx,
                          const descriptor_base& desc,
                          const infer_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_cpu<Float, method::by_default> {
    infer_result operator()(const context_cpu& ctx,
                            const descriptor_base& desc,
                            const infer_input& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::by_default>;
template struct infer_kernel_cpu<double, method::by_default>;

} // namespace oneapi::dal::kmeans::backend
