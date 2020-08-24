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

#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/algo/kmeans_init/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using std::int64_t;
using dal::backend::context_cpu;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop          = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_init_dense_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<daal_kmeans_init::defaultDense, Float, Cpu>;

template <typename Float>
static compute_result call_daal_kernel(const context_cpu& ctx,
                                       const descriptor_base& desc,
                                       const table& data) {
    const int64_t column_count  = data.get_column_count();
    const int64_t cluster_count = desc.get_cluster_count();

    daal_kmeans_init::Parameter par(cluster_count);

    auto arr_data          = row_accessor<const Float>{ data }.pull();
    const auto daal_data   = interop::convert_to_daal_homogen_table(arr_data,
                                                                  data.get_row_count(),
                                                                  data.get_column_count());
    const size_t len_input = 1;
    daal::data_management::NumericTable* input[len_input] = { daal_data.get() };

    array<Float> arr_centroids = array<Float>::empty(cluster_count * column_count);
    const auto daal_centroids =
        interop::convert_to_daal_homogen_table(arr_centroids, cluster_count, column_count);
    const size_t len_output                                 = 1;
    daal::data_management::NumericTable* output[len_output] = { daal_centroids.get() };

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_kmeans_init_dense_kernel_t>(ctx,
                                                                          len_input,
                                                                          input,
                                                                          len_output,
                                                                          output,
                                                                          &par,
                                                                          *(par.engine)));

    return compute_result().set_centroids(dal::detail::homogen_table_builder{}
                                              .reset(arr_centroids, cluster_count, column_count)
                                              .build());
}

template <typename Float>
static compute_result compute(const context_cpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense> {
    compute_result operator()(const context_cpu& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense>;
template struct compute_kernel_cpu<double, method::dense>;

} // namespace oneapi::dal::kmeans_init::backend
