/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

//for softmax calculation
#ifdef SOFTMAX
#include <thrust/reduce.h>
#endif


#ifdef SOFTMAX
struct exp_add {
    __host__ __device__
    int operator()(const float& x,const int& y) const {
        return x+expf(y);
    }
};
#endif

extern "C" bool
NvDsInferParseRTDETRCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

__global__ void decodeTensorRTDETRCuda(NvDsInferParseObjectInfo *binfo, const float* boxes, const float* scores, const float* classes,
    const int outputSize, const int netW, const int netH, const float minPreclusterThreshold, const float exp_sum)
{
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_id >= outputSize) {
        return;
    }
#ifdef SOFTMAX
    // do softmax calculation here
    float maxProb = expf(scores[x_id])/exp_sum;
#else
    float maxProb = scores[x_id];
#endif
    int maxIndex = (int) classes[x_id];

    if (maxProb < minPreclusterThreshold) {
        binfo[x_id].detectionConfidence = 0.0;
        return;
    }

    float bxc = boxes[x_id * 4 + 0];
    float byc = boxes[x_id * 4 + 1];
    float bw = boxes[x_id * 4 + 2];
    float bh = boxes[x_id * 4 + 3];

    float x0 = bxc - bw / 2;
    float y0 = byc - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = fminf(float(netW), fmaxf(float(0.0), x0));
    y0 = fminf(float(netH), fmaxf(float(0.0), y0));
    x1 = fminf(float(netW), fmaxf(float(0.0), x1));
    y1 = fminf(float(netH), fmaxf(float(0.0), y1));

    binfo[x_id].left = x0;
    binfo[x_id].top = y0;
    binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), x1 - x0));
    binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), y1 - y0));
    binfo[x_id].detectionConfidence = maxProb;
    binfo[x_id].classId = maxIndex;
}

static bool NvDsInferParseCustomRTDETRCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
    const NvDsInferLayerInfo& scores = outputLayersInfo[1];
    const NvDsInferLayerInfo& classes = outputLayersInfo[2];

    const int outputSize = boxes.inferDims.d[0];

    thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

    float minPreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
                                                      detectionParams.perClassPreclusterThreshold.end()));

    int threads_per_block = 1024;
    int number_of_blocks = ((outputSize - 1) / threads_per_block) + 1;
#ifdef SOFTMAX
    // do softmax calculation here
    // put scores on the device
    thrust::device_vector<float> raw_scores((float*) (scores.buffer),(float*) (scores.buffer)+outputSize);
    float exp_sum = thrust::reduce(raw_scores.begin(),raw_scores.end(),0,exp_add());
#else
    float exp_sum=1.;
#endif
    decodeTensorRTDETRCuda<<<number_of_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(objects.data()), (float*) (boxes.buffer), (float*) (scores.buffer),
    (float*) (classes.buffer), outputSize, networkInfo.width, networkInfo.height, minPreclusterThreshold,exp_sum);

    objectList.resize(outputSize);
    thrust::copy(objects.begin(), objects.end(), objectList.begin());

    return true;
}

extern "C" bool
NvDsInferParseRTDETRCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseCustomRTDETRCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRTDETRCuda);
