/* file: kernel_scheduler_sycl.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifdef DAAL_SYCL_INTERFACE
    #ifndef __DAAL_ONEAPI_INTERNAL_KERNEL_SCHEDULER_SYCL_H__
        #define __DAAL_ONEAPI_INTERNAL_KERNEL_SCHEDULER_SYCL_H__

        #include <CL/cl.h>
        #include <CL/sycl.hpp>

        #include "sycl/internal/daal_level_zero_common.h"

        #ifndef DAAL_DISABLE_LEVEL_ZERO
            #include "sycl/internal/daal_ze_module_helper.h"
        #endif // DAAL_DISABLE_LEVEL_ZERO

        #include <cstring>
        #include <vector>

        #include "sycl/internal/error_handling.h"
        #include "sycl/internal/execution_context.h"
        #include "sycl/internal/types_utils_cxx11.h"
        #include "services/daal_string.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
template <typename OpenClType, typename OpenClRetain, typename OpenClRelease>
class OpenClResourceRef : public Base
{
public:
    OpenClResourceRef() : _resource(nullptr) {}

    OpenClResourceRef(OpenClType & resource) : _resource(resource) {}

    OpenClResourceRef(const OpenClResourceRef & other)
    {
        _resource = other._resource;
        OpenClRetain()(_resource);
    }

    OpenClResourceRef(OpenClResourceRef && other) { _resource = other.release(); }

    ~OpenClResourceRef() { reset(); }

    operator bool() const { return get() == nullptr; }

    OpenClResourceRef & operator=(OpenClResourceRef other) { return swap(other); }

    OpenClResourceRef & operator=(OpenClResourceRef && other)
    {
        _resource = other.release();
        return *this;
    }

    OpenClResourceRef & swap(OpenClResourceRef & other)
    {
        OpenClType tmp  = _resource;
        _resource       = other._resource;
        other._resource = tmp;
        return *this;
    }

    OpenClType release()
    {
        OpenClType tmp = _resource;
        _resource      = nullptr;
        return tmp;
    }

    void reset(OpenClType resource = nullptr)
    {
        OpenClRelease()(_resource);
        _resource = resource;
    }

    OpenClType get() const { return _resource; }

private:
    OpenClType _resource;
};

        #define DAAL_DECLARE_OPENCL_OPERATOR(type_, name_) \
            struct OpenCl##name_                           \
            {                                              \
                void operator()(type_ p) { cl##name_(p); } \
            }

DAAL_DECLARE_OPENCL_OPERATOR(cl_program, RetainProgram);
DAAL_DECLARE_OPENCL_OPERATOR(cl_program, ReleaseProgram);
DAAL_DECLARE_OPENCL_OPERATOR(cl_kernel, RetainKernel);
DAAL_DECLARE_OPENCL_OPERATOR(cl_kernel, ReleaseKernel);
DAAL_DECLARE_OPENCL_OPERATOR(cl_context, RetainContext);
DAAL_DECLARE_OPENCL_OPERATOR(cl_context, ReleaseContext);
DAAL_DECLARE_OPENCL_OPERATOR(cl_device_id, RetainDevice);
DAAL_DECLARE_OPENCL_OPERATOR(cl_device_id, ReleaseDevice);

        #undef DAAL_DECLARE_OPENCL_OPERATOR

typedef OpenClResourceRef<cl_device_id, OpenClRetainDevice, OpenClReleaseDevice> OpenClDeviceRef;

class OpenClContextRef : public OpenClResourceRef<cl_context, OpenClRetainContext, OpenClReleaseContext>
{
public:
    OpenClContextRef() = default;

    explicit OpenClContextRef(cl_device_id clDevice, services::Status * status = nullptr) : _clDeviceRef(clDevice)
    {
        cl_int err = 0;
        reset(clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err));
        DAAL_CHECK_OPENCL(err, status)
    }

    OpenClDeviceRef getDeviceRef() { return _clDeviceRef; }

private:
    OpenClDeviceRef _clDeviceRef;
};

        #ifndef DAAL_DISABLE_LEVEL_ZERO
class LevelZeroOpenClInteropContext : public Base
{
public:
    LevelZeroOpenClInteropContext() = default;

    LevelZeroOpenClInteropContext(const LevelZeroOpenClInteropContext &) = delete;

    explicit LevelZeroOpenClInteropContext(cl::sycl::queue & deviceQueue, services::Status * status = nullptr) { reset(deviceQueue, status); }

    void reset(cl::sycl::queue & deviceQueue, services::Status * status = nullptr)
    {
        services::Status localStatus;
        cl_device_id clDevice;
        findDevice(&clDevice, deviceQueue.get_device().get_info<cl::sycl::info::device::vendor_id>(),
                   deviceQueue.get_device().get_info<cl::sycl::info::device::max_clock_frequency>(), &localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }
        _clDeviceRef.reset(clDevice);

        cl_int err = 0;
        _clContextRef.reset(clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err));
        DAAL_CHECK_OPENCL(err, status)
    }

    void findDevice(cl_device_id * pClDevice, unsigned int vendor_id, unsigned int frq, services::Status * status = nullptr)
    {
        constexpr auto maxPlatforms = 16;
        cl_platform_id platIds[maxPlatforms];
        cl_uint nplat, ndev;

        DAAL_CHECK_OPENCL(clGetPlatformIDs(maxPlatforms, platIds, &nplat), status);

        for (int pidx = 0; pidx < nplat && pidx < maxPlatforms; pidx++)
        {
            if (clGetDeviceIDs(platIds[pidx], CL_DEVICE_TYPE_GPU, 1, pClDevice, &ndev) == CL_SUCCESS)
            {
                if (ndev > 0)
                {
                    cl_uint dVid = 0;
                    clGetDeviceInfo(*pClDevice, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &dVid, NULL);

                    cl_uint dFrq = 0;
                    clGetDeviceInfo(*pClDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &dFrq, NULL);

                    if (dVid == vendor_id && dFrq == frq) return;
                }
            }
        }

        services::internal::tryAssignStatus(status, services::ErrorDeviceSupportNotImplemented);
    }

    OpenClDeviceRef & getOpenClDeviceRef() { return _clDeviceRef; }
    OpenClContextRef & getOpenClContextRef() { return _clContextRef; }

private:
    OpenClContextRef _clContextRef;
    OpenClDeviceRef _clDeviceRef;
};
        #endif // DAAL_DISABLE_LEVEL_ZERO

class OpenClProgramRef : public OpenClResourceRef<cl_program, OpenClRetainProgram, OpenClReleaseProgram>
{
public:
    OpenClProgramRef() {}

    explicit OpenClProgramRef(cl_context clContext, cl_device_id clDevice, const char * programName, const char * programSrc, const char * options,
                              services::Status * status = nullptr)
    {
        initOpenClProgramRef(clContext, clDevice, programName, programSrc, options, status);
    }
        #ifndef DAAL_DISABLE_LEVEL_ZERO
    explicit OpenClProgramRef(cl_context clContext, cl_device_id clDevice, cl::sycl::queue & deviceQueue, const char * programName,
                              const char * programSrc, const char * options, services::Status * status = nullptr)
    {
        services::Status localStatus;
        initOpenClProgramRef(clContext, clDevice, programName, programSrc, options, &localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }
        initModuleLevelZero(deviceQueue, status);
    }

    cl::sycl::program getProgramLevelZero() const { return _moduleLevelZeroPtr->getZeProgram(); }
        #endif // DAAL_DISABLE_LEVEL_ZERO

    const char * getName() const { return _programName.c_str(); }

private:
    void initOpenClProgramRef(cl_context clContext, cl_device_id clDevice, const char * programName, const char * programSrc, const char * options,
                              services::Status * status = nullptr)
    {
        _programName           = programName;
        cl_int err             = 0;
        const char * sources[] = { programSrc };
        const size_t lengths[] = { std::strlen(programSrc) };
        reset(clCreateProgramWithSource(clContext, 1, sources, lengths, &err));
        DAAL_CHECK_OPENCL(err, status)

        err = clBuildProgram(get(), 1, &clDevice, options, nullptr, nullptr);

        #ifdef DAAL_EXECUTION_CONTEXT_VERBOSE
        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t logLen = 0;
            clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
            services::Collection<char> buildLogCollection(logLen);
            char * buildLog = buildLogCollection.data();
            if (buildLog == nullptr)
            {
                printf("common_ocl.ocdBuildProgramFromFile() - Heap Overflow! Cannot "
                       "allocate space for buildLog.");
                DAAL_CHECK_OPENCL(err, status)
            }
            clGetProgramBuildInfo(get(), clDevice, CL_PROGRAM_BUILD_LOG, logLen, (void *)buildLog, nullptr);
            printf("CL Error %d: Failed to build program! Log:\n%s", err, buildLog);
        }
        #endif
        DAAL_CHECK_OPENCL(err, status)
    }

        #ifndef DAAL_DISABLE_LEVEL_ZERO
    void initModuleLevelZero(cl::sycl::queue & deviceQueue, services::Status * status = nullptr)
    {
        size_t binarySize = 0;
        DAAL_CHECK_OPENCL(clGetProgramInfo(get(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL), status);

        auto binary = (unsigned char *)daal::services::daal_malloc(binarySize);
        if (binary == nullptr)
        {
            services::internal::tryAssignStatus(status, services::ErrorMemoryAllocationFailed);
            return;
        }

        DAAL_CHECK_OPENCL(clGetProgramInfo(get(), CL_PROGRAM_BINARIES, sizeof(binary), &binary, NULL), status);
        _moduleLevelZeroPtr.reset(new ZeModuleHelper(deviceQueue, binarySize, binary, status));
        daal::services::daal_free(binary);
    }
        #endif // DAAL_DISABLE_LEVEL_ZERO

private:
    services::String _programName;
        #ifndef DAAL_DISABLE_LEVEL_ZERO
    ZeModuleHelperPtr _moduleLevelZeroPtr;
        #endif // DAAL_DISABLE_LEVEL_ZERO
};

class OpenClKernelRef : public OpenClResourceRef<cl_kernel, OpenClRetainKernel, OpenClReleaseKernel>
{
public:
    OpenClKernelRef() = default;

    explicit OpenClKernelRef(cl_program clProgram, const char * kernelName, services::Status * status = nullptr)
    {
        cl_int err = 0;
        reset(clCreateKernel(clProgram, kernelName, &err));
        DAAL_CHECK_OPENCL(err, status)
    }
};

        #ifndef DAAL_DISABLE_LEVEL_ZERO
class OpenClKernelLevelZeroRef : public Base
{
public:
    OpenClKernelLevelZeroRef() = default;

    explicit OpenClKernelLevelZeroRef(const char * kernelName, services::Status * status = nullptr) : _kernelName(kernelName) {}

    const char * getName() const { return _kernelName.c_str(); }

private:
    services::String _kernelName;
};
        #endif // DAAL_DISABLE_LEVEL_ZERO

class OpenClKernel : public Base, public KernelIface
{
public:
    explicit OpenClKernel(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef)
        : _executionTarget(executionTarget), _clProgramRef(programRef)
    {}

    void schedule(KernelSchedulerIface & scheduler, const KernelRange & range, const KernelArguments & args,
                  services::Status * status = nullptr) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    void schedule(KernelSchedulerIface & scheduler, const KernelNDRange & range, const KernelArguments & args,
                  services::Status * status = nullptr) const DAAL_C11_OVERRIDE
    {
        scheduler.schedule(*this, range, args, status);
    }

    ExecutionTargetId getTarget() const { return _executionTarget; }

    virtual cl::sycl::kernel toSycl(const cl::sycl::context & ctx) const = 0;

    const OpenClProgramRef & getProgramRef() const { return _clProgramRef; }

private:
    OpenClProgramRef _clProgramRef;
    ExecutionTargetId _executionTarget;
};

class OpenClKernelNative : public OpenClKernel
{
public:
    explicit OpenClKernelNative(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef, const OpenClKernelRef & kernelRef)
        : OpenClKernel(executionTarget, programRef), _clKernelRef(kernelRef)
    {}

    cl::sycl::kernel toSycl(const cl::sycl::context & ctx) const { return cl::sycl::kernel(_clKernelRef.get(), ctx); }

private:
    OpenClKernelRef _clKernelRef;
};

        #ifndef DAAL_DISABLE_LEVEL_ZERO
class OpenClKernelLevelZero : public OpenClKernel
{
public:
    explicit OpenClKernelLevelZero(ExecutionTargetId executionTarget, const OpenClProgramRef & programRef, const OpenClKernelLevelZeroRef & kernelRef)
        : OpenClKernel(executionTarget, programRef),
          _clKernelRef(kernelRef),
          syclKernel(getProgramRef().getProgramLevelZero().get_kernel(_clKernelRef.getName()))
    {}

    cl::sycl::kernel toSycl(const cl::sycl::context & ctx) const { return syclKernel; }

private:
    OpenClKernelLevelZeroRef _clKernelRef;
    cl::sycl::kernel syclKernel;
};
        #endif // DAAL_DISABLE_LEVEL_ZERO

class SyclBufferStorage
{
public:
    template <typename T>
    void add(const cl::sycl::buffer<T, 1> & buffer)
    {
        _buffers.push_back(services::internal::Any(buffer));
    }

private:
    std::vector<services::internal::Any> _buffers;
};

class SyclKernelSchedulerArgHandler
{
public:
    SyclKernelSchedulerArgHandler(cl::sycl::handler & handler, SyclBufferStorage & storage, size_t argumentIndex, const KernelArgument & arg)
        : _handler(handler), _storage(storage), _argumentIndex(argumentIndex), _argument(arg)
    {}

    template <typename T>
    void operator()(Typelist<T>)
    {
        switch (_argument.argType())
        {
        case KernelArgumentTypes::publicBuffer: return handlePublicBuffer<T>();
        case KernelArgumentTypes::privateBuffer: return handlePrivateBuffer<T>();
        case KernelArgumentTypes::publicConstant: return handlePublicConstant<T>();
        }

        DAAL_ASSERT(!"Unexpected kernel argument type");
    }

private:
    template <typename T>
    void handlePublicBuffer()
    {
        auto buffer = _argument.get<services::Buffer<T> >().toSycl();

        // Note: we need this storage to keep all sycl buffers alive
        // while the kernel is running
        _storage.add(buffer);

        switch (_argument.accessMode())
        {
        case AccessModeIds::read: return handlePublicBuffer<cl::sycl::access::mode::read>(buffer);

        case AccessModeIds::write: return handlePublicBuffer<cl::sycl::access::mode::write>(buffer);

        case AccessModeIds::readwrite: return handlePublicBuffer<cl::sycl::access::mode::read_write>(buffer);
        }

        DAAL_ASSERT(!"Unexpected buffer access mode");
    }

    template <cl::sycl::access::mode mode, typename Buffer>
    void handlePublicBuffer(Buffer & buffer)
    {
        auto accessor = buffer.template get_access<mode>(_handler);
        _handler.set_arg((int)_argumentIndex, accessor);
    }

    template <typename T>
    void handlePrivateBuffer()
    {
        DAAL_ASSERT(!"Local buffers are not supported");
    }

    template <typename T>
    void handlePublicConstant()
    {
        // XXX: Expression _cgh.set_arg(_idx, _arg.get<T>()) does not compile due
        // to bug in Intel SYCL implementation
        T value = _argument.get<T>();
        _handler.set_arg((int)_argumentIndex, value);
    }

    cl::sycl::handler & _handler;
    SyclBufferStorage & _storage;
    size_t _argumentIndex;
    const KernelArgument & _argument;
    services::Status _status;
};

template <int dim>
inline cl::sycl::range<dim> convertToSyclRange(const KernelRange &);

template <>
inline cl::sycl::range<1> convertToSyclRange<1>(const KernelRange & r)
{
    return cl::sycl::range<1>(r.upper1());
}

template <>
inline cl::sycl::range<2> convertToSyclRange<2>(const KernelRange & r)
{
        #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
    return cl::sycl::range<2>(r.upper2(), r.upper1());
        #else
    return cl::sycl::range<2>(r.upper1(), r.upper2());
        #endif
}

template <>
inline cl::sycl::range<3> convertToSyclRange<3>(const KernelRange & r)
{
        #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
    return cl::sycl::range<3>(r.upper3(), r.upper2(), r.upper1());
        #else
    return cl::sycl::range<3>(r.upper1(), r.upper2(), r.upper3());
        #endif
}

template <int dim>
inline cl::sycl::nd_range<dim> convertToSyclRange(const KernelNDRange &);

template <>
inline cl::sycl::nd_range<1> convertToSyclRange<1>(const KernelNDRange & r)
{
    return cl::sycl::nd_range<1>(cl::sycl::range<1>(r.global().upper1()), cl::sycl::range<1>(r.local().upper1()));
}

template <>
inline cl::sycl::nd_range<2> convertToSyclRange<2>(const KernelNDRange & r)
{
    return cl::sycl::nd_range<2>(
        #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
        cl::sycl::range<2>(r.global().upper2(), r.global().upper1()), cl::sycl::range<2>(r.local().upper2(), r.local().upper1())
        #else
        cl::sycl::range<2>(r.global().upper1(), r.global().upper2()), cl::sycl::range<2>(r.local().upper1(), r.local().upper2())
        #endif
    );
}

template <>
inline cl::sycl::nd_range<3> convertToSyclRange<3>(const KernelNDRange & r)
{
    return cl::sycl::nd_range<3>(
        #ifdef DAAL_SYCL_INTERFACE_REVERSED_RANGE
        cl::sycl::range<3>(r.global().upper3(), r.global().upper2(), r.global().upper1()),
        cl::sycl::range<3>(r.local().upper3(), r.local().upper2(), r.local().upper1())
        #else
        cl::sycl::range<3>(r.global().upper1(), r.global().upper2(), r.global().upper3()),
        cl::sycl::range<3>(r.local().upper1(), r.local().upper2(), r.local().upper3())
        #endif
    );
}

class SyclKernelScheduler : public Base, public KernelSchedulerIface
{
public:
    explicit SyclKernelScheduler(cl::sycl::queue & deviceQueue) : _queue(deviceQueue) {}

    void schedule(const OpenClKernel & kernel, const KernelRange & range, const KernelArguments & args,
                  services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        scheduleImplSafe(range, kernel, args, status);
    }

    void schedule(const OpenClKernel & kernel, const KernelNDRange & range, const KernelArguments & args,
                  services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        scheduleImplSafe(range, kernel, args, status);
    }

private:
    template <typename Range>
    void scheduleImplSafe(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, services::Status * status = nullptr)
    {
        try
        {
            scheduleImpl(range, kernel, args, status);
        }
        catch (const cl::sycl::exception & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }

    template <typename Range>
    void scheduleImpl(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, services::Status * status = nullptr)
    {
        switch (kernel.getTarget())
        {
        case ExecutionTargetIds::device: return scheduleOnDevice(range, kernel, args, status);

        case ExecutionTargetIds::host: return services::internal::tryAssignStatus(status, services::ErrorID::ErrorMethodNotImplemented);
        }

        DAAL_ASSERT(!"Unexpected execution target");
    }

    template <typename Range>
    void scheduleOnDevice(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, services::Status * status = nullptr)
    {
        switch (range.dimensions())
        {
        case 1: return scheduleSycl(convertToSyclRange<1>(range), kernel, args, status);
        case 2: return scheduleSycl(convertToSyclRange<2>(range), kernel, args, status);
        case 3: return scheduleSycl(convertToSyclRange<3>(range), kernel, args, status);
        }

        DAAL_ASSERT(!"Unexpected number of dimensions");
    }

    template <typename Range>
    void scheduleSycl(const Range & range, const OpenClKernel & kernel, const KernelArguments & args, services::Status * status = nullptr)
    {
        SyclBufferStorage bufferStorage;

        cl::sycl::kernel syclKernel = kernel.toSycl(_queue.get_context());

        auto event = _queue.submit([&](cl::sycl::handler & cgh) {
            passArguments(cgh, bufferStorage, args);
            cgh.parallel_for(range, syclKernel);
        });

        event.wait_and_throw();
    }

    void passArguments(cl::sycl::handler & cgh, SyclBufferStorage & storage, const KernelArguments & args) const
    {
        for (size_t i = 0; i < args.size(); i++)
        {
            const auto & arg = args.get(i);
            SyclKernelSchedulerArgHandler argHandler(cgh, storage, i, arg);
            TypeDispatcher::dispatch(arg.dataType(), argHandler);
        }
    }

private:
    cl::sycl::queue & _queue;
};

} // namespace interface1
} // namespace internal
} // namespace oneapi
} // namespace daal

    #endif
#endif // DAAL_SYCL_INTERFACE
