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

#include "oneapi/dal/common.hpp"

namespace oneapi::dal::backend {

void convert_vector(const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dest_type,
                    std::int64_t size);

void convert_vector(const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dest_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t size);

} // namespace oneapi::dal::backend
