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

#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::detail {

class table_builder_impl_iface : public access_provider_iface {
public:
    virtual table build() = 0;
};

class homogen_table_builder_iface : public table_builder_impl_iface {
public:
    virtual homogen_table build_homogen() = 0;
    virtual void reset(homogen_table&& t) = 0;
    virtual void reset(const array<byte_t>& data,
                       std::int64_t row_count,
                       std::int64_t column_count) = 0;
    virtual void set_data_type(data_type dt) = 0;
    virtual void set_feature_type(feature_type ft) = 0;
    virtual void allocate(std::int64_t row_count, std::int64_t column_count) = 0;
    virtual void set_layout(data_layout layout) = 0;
    virtual void copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) = 0;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual void allocate(const sycl::queue& queue,
                          std::int64_t row_count,
                          std::int64_t column_count,
                          sycl::usm::alloc kind) = 0;
    virtual void copy_data(sycl::queue& queue,
                           const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;
#endif
};

template <typename Impl>
class table_builder_impl_wrapper : public table_builder_impl_iface, public base {
public:
#ifdef ONEAPI_DAL_DATA_PARALLEL
    table_builder_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }),
              dpc_access_ptr_(new access_wrapper_dpc<Impl>{ impl_ }) {}
#else
    table_builder_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }) {}
#endif

    virtual table build() override {
        return impl_.build();
    }

    virtual access_iface_host& get_access_iface_host() const override {
        return *host_access_ptr_.get();
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual access_iface_dpc& get_access_iface_dpc() const override {
        return *dpc_access_ptr_.get();
    }
#endif

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;

    unique<access_iface_host> host_access_ptr_;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    unique<access_iface_dpc> dpc_access_ptr_;
#endif
};

template <typename Impl>
class homogen_table_builder_impl_wrapper : public homogen_table_builder_iface, public base {
public:
#ifdef ONEAPI_DAL_DATA_PARALLEL
    homogen_table_builder_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }),
              dpc_access_ptr_(new access_wrapper_dpc<Impl>{ impl_ }) {}
#else
    homogen_table_builder_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }) {}
#endif

    virtual table build() override {
        return impl_.build();
    }

    virtual homogen_table build_homogen() override {
        return impl_.build();
    }

    virtual void reset(homogen_table&& t) override {
        impl_.reset(std::move(t));
    }
    virtual void reset(const array<byte_t>& data,
                       std::int64_t row_count,
                       std::int64_t column_count) override {
        impl_.reset(data, row_count, column_count);
    }
    virtual void set_data_type(data_type dt) override {
        impl_.set_data_type(dt);
    }
    virtual void set_feature_type(feature_type ft) override {
        impl_.set_feature_type(ft);
    }
    virtual void allocate(std::int64_t row_count, std::int64_t column_count) override {
        impl_.allocate(row_count, column_count);
    }
    virtual void set_layout(data_layout layout) override {
        impl_.set_layout(layout);
    }
    virtual void copy_data(const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) override {
        impl_.copy_data(data, row_count, column_count);
    }

    virtual access_iface_host& get_access_iface_host() const override {
        return *host_access_ptr_.get();
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual void allocate(const sycl::queue& queue,
                          std::int64_t row_count,
                          std::int64_t column_count,
                          sycl::usm::alloc kind) override {
        impl_.allocate(queue, row_count, column_count, kind);
    }
    virtual void copy_data(sycl::queue& queue,
                           const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) override {
        impl_.copy_data(queue, data, row_count, column_count);
    }

    virtual access_iface_dpc& get_access_iface_dpc() const override {
        return *dpc_access_ptr_.get();
    }
#endif

private:
    Impl impl_;

    unique<access_iface_host> host_access_ptr_;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    unique<access_iface_dpc> dpc_access_ptr_;
#endif
};

} // namespace oneapi::dal::detail
