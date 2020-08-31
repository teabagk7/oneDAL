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

#include <new>
#include <stdexcept>
#include <system_error>

#include "oneapi/dal/common.hpp"

namespace oneapi::dal {

class ONEAPI_DAL_EXPORT exception {
public:
    virtual ~exception() = default;
    virtual const char* what() const noexcept = 0;
};

class ONEAPI_DAL_EXPORT logic_error : public exception {};
class ONEAPI_DAL_EXPORT runtime_error : public exception {};

class ONEAPI_DAL_EXPORT invalid_argument : public logic_error, public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT domain_error : public logic_error, public std::domain_error {
public:
    using std::domain_error::domain_error;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT out_of_range : public logic_error, public std::out_of_range {
public:
    using std::out_of_range::out_of_range;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT unimplemented_error : public logic_error, public std::logic_error {
public:
    using std::logic_error::logic_error;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT unavailable_error : public logic_error, public std::logic_error {
public:
    using std::logic_error::logic_error;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT range_error : public runtime_error, public std::range_error {
public:
    using std::range_error::range_error;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT system_error : public runtime_error, public std::system_error {
public:
    using std::system_error::system_error;
    const char* what() const noexcept override;
    const std::error_code& code() const noexcept;
};

class ONEAPI_DAL_EXPORT internal_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override;
};

class ONEAPI_DAL_EXPORT bad_alloc : public exception, public std::bad_alloc {
public:
    bad_alloc() noexcept = default;
    const char* what() const noexcept override;
};

} // namespace oneapi::dal
