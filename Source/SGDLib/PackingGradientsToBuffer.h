//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include "Criterion.h"
#include "DataReaderHelpers.h"
#include "Globals.h"
#include "ProgressTracing.h"

#include <set>
#include <string>
#include <vector>

#define _DEFAULT_PACK_THRESHOLD_SIZE (32 * 1024)

template <class ElemType>
int PackGradientsToBuffer(const std::vector<Matrix<ElemType>*>& gradients, std::unique_ptr<Matrix<ElemType>> AggregationBuffer, std::unique_ptr<std::unordered_set<size_t>> PackedGradientsIndex)
{
    int ret = 0;
    int deviceId = gradients[0]->GetDeviceId();
    size_t numGradGradients = gradients.size();

    PackedGradientsIndex.reset();
    AggregationBuffer.reset();

    size_t PackedGradientsSizeInElements = 0;
    for (size_t i = 0; i < numGradGradients; ++i)
    {
        if ((sizeof(ElemType) * gradients[i]->GetNumElements()) <= packThresholdSize)
        {
            PackedGradientsSizeInElements += gradients[i]->GetNumElements;
            PackedGradientsIndex->insert(i);
        }
    }

    if (PackedGradientsSizeInElements > 0)
    {
        AggregationBuffer.reset(new (std::nothrow) Matrix<ElemType>(1, totalGradientsSizeInElements, deviceId));

        if (AggregationBuffer == 0)
        {
            PackedGradientsIndex.reset();
            AggregationBuffer.reset();
            return -1;
        }
        else
        {
            size_t offset = 0;
            for (size_t i : *PackedGradientsIndex)
            {
                AggregationBuffer->ColumnSlice(offset, gradients[i]->GetNumElements()).AssignValuesOf(gradients[i]->Reshaped(1, gradients[i]->GetNumElements()));
                offset += gradients[i]->GetNumElements();
            }
        }
    }
    return 1;
}