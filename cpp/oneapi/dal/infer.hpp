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

#include "oneapi/dal/detail/infer_ops.hpp"

namespace oneapi::dal {

template <typename... Args>
auto infer(Args&&... args) {
    return detail::infer_dispatch(std::forward<Args>(args)...);
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename... Args>
auto infer(sycl::queue& queue, Args&&... args) {
    return detail::infer_dispatch(detail::data_parallel_policy{ queue },
                                  std::forward<Args>(args)...);
}
#endif

} // namespace oneapi::dal
