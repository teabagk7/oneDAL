/* file: kmeans_lloyd_helper.h */
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

/*
//++
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#ifndef _KMEANS_LLOYD_HELPER_H__
#define _KMEANS_LLOYD_HELPER_H__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_defines.h"
#include "algorithms/kernel/service_error_handling.h"

#include "algorithms/threading/threading.h"
#include "externals/service_blas.h"
#include "externals/service_spblas.h"
#include "service/kernel/service_data_utils.h"
#include "service/service_environment.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct TlsTask
{
    DAAL_NEW_DELETE();

    TlsTask(int dim, int clNum, int maxBlockSize)
    {
        mklBuff  = service_scalable_calloc<algorithmFPType, cpu>(maxBlockSize * clNum);
        cS1      = service_scalable_calloc<algorithmFPType, cpu>(clNum * dim);
        cS0      = service_scalable_calloc<int, cpu>(clNum);
        cValues  = service_scalable_calloc<algorithmFPType, cpu>(clNum);
        cIndices = service_scalable_calloc<size_t, cpu>(clNum);
    }

    ~TlsTask()
    {
        if (mklBuff)
        {
            service_scalable_free<algorithmFPType, cpu>(mklBuff);
        }
        if (cS1)
        {
            service_scalable_free<algorithmFPType, cpu>(cS1);
        }
        if (cS0)
        {
            service_scalable_free<int, cpu>(cS0);
        }
        if (cValues)
        {
            service_scalable_free<algorithmFPType, cpu>(cValues);
        }
        if (cIndices)
        {
            service_scalable_free<size_t, cpu>(cIndices);
        }
    }

    static TlsTask<algorithmFPType, cpu> * create(const size_t dim, const size_t clNum, const size_t maxBlockSize)
    {
        TlsTask<algorithmFPType, cpu> * result = new TlsTask<algorithmFPType, cpu>(dim, clNum, maxBlockSize);
        if (!result)
        {
            return nullptr;
        }
        if (!result->mklBuff || !result->cS1 || !result->cS0)
        {
            delete result;
            return nullptr;
        }
        return result;
    }

    algorithmFPType * mklBuff = nullptr;
    algorithmFPType * cS1     = nullptr;
    int * cS0                 = nullptr;
    algorithmFPType goalFunc  = 0.0;
    size_t cNum               = 0;
    algorithmFPType * cValues = nullptr;
    size_t * cIndices         = nullptr;
};
template <typename algorithmFPtype, CpuType cpu>
struct BSHelper;
template <CpuType cpu>
struct BSHelper<double, cpu>
{
    static size_t kmeansGetBlockSize(const size_t nRows, const size_t dim, const size_t clNum)
    {
        size_t rows_fit_L1 = (getL1CacheSize() / 8 - (clNum * dim)) / (clNum + dim) * 0.8;
        size_t rows_fit_L2 = (getL2CacheSize() / 8 - (clNum * dim)) / (clNum + dim) * 0.8;
        size_t blockSize   = 96;

        if (rows_fit_L1 >= 256 && rows_fit_L1 <= 512)
        {
            blockSize = int(rows_fit_L1 / 16) * 16;
        }
        else if (rows_fit_L2 >= 8 && rows_fit_L2 <= 512)
        {
            blockSize = int(rows_fit_L2 / 8) * 8;
        }
        else if (rows_fit_L2 >= 512)
        {
            blockSize = 496;
        }

        blockSize = int(rows / (int(rows / blockSize / 56) * 56));

        return blockSize;
    }
};

template <CpuType cpu>
struct BSHelper<float, cpu>
{
    static size_t kmeansGetBlockSize(const size_t nRows, const size_t dim, const size_t clNum)
    {
        size_t rows_fit_L1 = (getL1CacheSize() / 4 - (clNum * dim)) / (clNum + dim) * 0.8;
        size_t rows_fit_L2 = (getL2CacheSize() / 4 - (clNum * dim)) / (clNum + dim) * 0.8;
        size_t blockSize   = 96;

        if (rows_fit_L1 >= 256 * 2 && rows_fit_L1 <= 512 * 2)
        {
            blockSize = int(rows_fit_L1 / 16) * 16;
        }
        else if (rows_fit_L2 >= 8 && rows_fit_L2 <= 512 * 2)
        {
            blockSize = int(rows_fit_L2 / 8) * 8;
        }
        else if (rows_fit_L2 >= 512 * 2)
        {
            blockSize = 496 * 2;
        }
        blockSize = int(rows / (int(rows / blockSize / 56) * 56));

        return blockSize;
    }
};
template <typename algorithmFPType>
struct Fp2IntSize
{};
template <>
struct Fp2IntSize<float>
{
    typedef int IntT;
};
template <>
struct Fp2IntSize<double>
{
    typedef __int64 IntT;
};

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
