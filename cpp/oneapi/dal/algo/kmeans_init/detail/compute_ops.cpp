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

#include "oneapi/dal/algo/kmeans_init/detail/compute_ops.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::kmeans_init::detail {
using oneapi::dal::detail::host_policy;

template <typename Float, typename Method>
struct ONEAPI_DAL_EXPORT compute_ops_dispatcher<host_policy, Float, Method> {
    compute_result operator()(const host_policy& ctx,
                              const descriptor_base& desc,
                              const compute_input& input) const {
        using kernel_dispatcher_t =
            dal::backend::kernel_dispatcher<backend::compute_kernel_cpu<Float, Method>>;
        return kernel_dispatcher_t()(ctx, desc, input);
    }
};

#define INSTANTIATE(F, M) \
    template struct ONEAPI_DAL_EXPORT compute_ops_dispatcher<host_policy, F, M>;

INSTANTIATE(float, method::dense)
INSTANTIATE(double, method::dense)
INSTANTIATE(float, method::random_dense)
INSTANTIATE(double, method::random_dense)
INSTANTIATE(float, method::plus_plus_dense)
INSTANTIATE(double, method::plus_plus_dense)
INSTANTIATE(float, method::parallel_plus_dense)
INSTANTIATE(double, method::parallel_plus_dense)

} // namespace oneapi::dal::kmeans_init::detail
