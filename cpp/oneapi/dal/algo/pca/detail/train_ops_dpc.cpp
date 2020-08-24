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

#include "oneapi/dal/algo/pca/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/pca/detail/train_ops.hpp"
#include "oneapi/dal/backend/dispatcher_dpc.hpp"

namespace oneapi::dal::pca::detail {
using oneapi::dal::detail::data_parallel_policy;

template <typename Float, typename Method>
struct ONEAPI_DAL_EXPORT train_ops_dispatcher<data_parallel_policy, Float, Method> {
    train_result operator()(const data_parallel_policy& ctx,
                            const descriptor_base& params,
                            const train_input& input) const {
        using kernel_dispatcher_t =
            dal::backend::kernel_dispatcher<backend::train_kernel_cpu<Float, Method>,
                                            backend::train_kernel_gpu<Float, Method>>;
        return kernel_dispatcher_t{}(ctx, params, input);
    }
};

#define INSTANTIATE(F, M) \
    template struct ONEAPI_DAL_EXPORT train_ops_dispatcher<data_parallel_policy, F, M>;

INSTANTIATE(float, method::cov)
INSTANTIATE(float, method::svd)
INSTANTIATE(double, method::cov)
INSTANTIATE(double, method::svd)

} // namespace oneapi::dal::pca::detail
