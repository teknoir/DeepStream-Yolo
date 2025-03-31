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

#include "nvdsinfer_custom_impl.h"

#include "utils.h"
#include <cmath>
extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);
  //if (maxIndex==0){
  //  std::cout << "maxIndex: " << maxIndex << std::endl;
  //  std::cout << "bbi.width: " << bbi.width << std::endl;
  //  std::cout << "bbi.height: " << bbi.height << std::endl;
  //  std::cout << "bbi.left: " << bbi.left << std::endl;
  //  std::cout << "bbi.top: " << bbi.top << std::endl;
  //}
  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYoloE(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = boxes[b * 4 + 0];
    float by1 = boxes[b * 4 + 1];
    float bx2 = boxes[b * 4 + 2];
    float by2 = boxes[b * 4 + 3];
    



    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;
  
  //std::cout << "calling decodeTensorYolo\n";
  //std::cout << "outputSize: " << outputSize << "\n";
  //std::cout << "netW: " << netW << "\n";
  //std::cout << "netH: " << netH << "\n";
  
  //std::cout << "preclusterThreshold: " << preclusterThreshold.size() << "\n";
  
  int count = 0;
  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];
    if (maxProb < preclusterThreshold[maxIndex]) {
        continue;
      }
    
    #ifdef XYWHC
      //std::cout << "using XYWHC" << std::endl;
     float bxc = boxes[b * 4 + 0];
     float byc = boxes[b * 4 + 1];
     float bw = boxes[b * 4 + 2];
     float bh = boxes[b * 4 + 3];
     //std::cout << "bx[0] " << boxes[b*4] << " bx[1] " << boxes[b*4+1]  << " bx[2] " << boxes[b*4+2]  << " bx[3] " << boxes[b*4+3]   << std::endl;
     float bx1 = bxc - bw / 2;
     float by1 = byc - bh / 2;
     float bx2 = bx1 + bw;
     float by2 = by1 + bh;
    #elif XYWH
     //std::cout << "using XYWH" << std::endl;
     float bx1 = boxes[b * 4 + 0];
     float by1 = boxes[b * 4 + 1];
     float bw = boxes[b * 4 + 2];
     float bh = boxes[b * 4 + 3];
     float bx2 = bx1 + bw;
     float by2 = by1 + bh;
    #else
     //std::cout << "using XYXY" << std::endl;
     float bx1 = boxes[b * 4 + 0];
     float by1 = boxes[b * 4 + 1];
     float bx2 = boxes[b * 4 + 2];
     float by2 = boxes[b * 4 + 3];
    #endif
    /*
    if(maxIndex==0){
    std::cout << "score " << maxProb <<std::endl;
    std::cout << "class " << maxIndex <<std::endl;
    NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);
    std::cout << "bbi.width: " << bbi.width << std::endl; 
    std::cout << "bbi.height: " << bbi.height << std::endl;
    std::cout << "bbi.left: " << bbi.left << std::endl;
    std::cout << "bbi.top: " << bbi.top << std::endl;
    std::cout << "bx1 " << bx1 << " by1 " << by1 << " bx2 " << bx2 << " by2 " << by2  << std::endl;
    std::cout << "netW " << netW << " netH " << netH << std::endl;
    std::cout << "bx[0] " << boxes[b*4] << " bx[1] " << boxes[b*4+1]  << " bx[2] " << boxes[b*4+2]  << " bx[3] " << boxes[b*4+3]   << std::endl;
    }
    */
    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
    count++;
  }
  //std::cout << "count: " << count << "\n";
  //std::cout << "binfo size: " << binfo.size() << "\n";
  return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorRFDETR(const float* boxes, const float* labels, const uint& outputSize, const uint& numClasses, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) 
  {
    uint maxIndex = 0;
    float maxProb = labels[b * numClasses];
    
    for (uint c = 0; c < numClasses; ++c)
    {
      if (labels[b * numClasses + c] > maxProb)
      {
        maxProb = labels[b * numClasses + c];
        maxIndex = c;
      }
    }
    /*
    if (maxProb < preclusterThreshold[maxIndex])
    {
      continue;
    }
    */

    float xc = boxes[b * 4 + 0]*(float)netW  ;
    float yc = boxes[b * 4 + 1]*(float)netH;
    float w = boxes[b * 4 + 2]*(float)netW;
    float h = boxes[b * 4 + 3]*(float)netH;
    float bx1 = xc - w/2;
    float by1 = yc - h/2;
    float bx2 = bx1 + w;
    float by2 = by1 + h;
    //std::cout << "--------------------------------" << std::endl;
    //std::cout << "bx1 " << bx1 << " by1 " << by1 << " bx2 " << bx2 << " by2 " << by2 << std::endl;
    // std::cout << "maxIndex " << maxIndex << " maxProb " << maxProb << " softmax " << softmax << std::endl;
    //std::cout << "--------------------------------" << std::endl;
    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }
  return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  auto layerFinder = [&outputLayersInfo](const std::string &name) -> const NvDsInferLayerInfo *{
      for (auto &layer : outputLayersInfo) {
        if (layer.dataType == FLOAT &&
            (layer.layerName && name == layer.layerName)) {
          //std::cout << "found layer name: " << layer.layerName << "\n";
          return &layer;
        }
      }
      return nullptr;
  };


  // need different parsing for RFDETR 
  #ifdef RFDETR

  const NvDsInferLayerInfo *boxes = layerFinder("dets");
  const NvDsInferLayerInfo *labels = layerFinder("labels");
  const uint outputSize = boxes->inferDims.d[0];
  const uint numClasses = labels->inferDims.d[1];
  //std::cout << "boxes number of dims: " << boxes->inferDims.numDims << "\n";
  //std::cout << "boxes number of elements dim 0: " << boxes->inferDims.d[0] << "\n";
  //std::cout << "boxes number of elements dim 1: " << boxes->inferDims.d[1] << "\n";
  //std::cout << "labels number of dims: " << labels->inferDims.numDims << "\n";
  //std::cout << "labels number of elements dim 0: " << labels->inferDims.d[0] << "\n";
  //std::cout << "labels number of elements dim 1: " << labels->inferDims.d[1] << "\n";
  
  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorRFDETR((const float*) (boxes->buffer),
      (const float*) (labels->buffer), outputSize, numClasses, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  #else
  const NvDsInferLayerInfo *boxes = layerFinder("boxes");
  const NvDsInferLayerInfo *scores = layerFinder("scores");
  const NvDsInferLayerInfo *classes = layerFinder("classes");
  const uint outputSize = boxes->inferDims.d[0];
  //std::cout << "boxes number of dims: " << boxes->inferDims.numDims << "\n";
  //std::cout << "boxes number of elements dim 0: " << boxes->inferDims.d[0] << "\n";
  //std::cout << "boxes number of elements dim 1: " << boxes->inferDims.d[1] << "\n";
  
  //std::cout << "calling decodeTensorYolo\n";
  //std::cout << "numClassesConfigured: " << detectionParams.numClassesConfigured << " - outputSize: " << outputSize << "\n";
  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo((const float*) (boxes->buffer),
      (const float*) (scores->buffer), (const float*) (classes->buffer), outputSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);
  //std::cout << "calling decodeTensorYolo - success\n";
  //std::cout << "outObjs size: " << outObjs.size() << "\n";
  
  #endif
  objects.insert(objects.end(), outObjs.begin(), outObjs.end());
  objectList = objects;
  //std::cout << "returning from NvDsInferParseCustomYolo\n";

  return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  //d::cout << "returning from NvDsInferParseYolo\n";
  //std::cout << "objectList size: " << objectList.size() << "\n";
  bool result = NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
  //std::cout << "objectList size: " << objectList.size() << "\n";
  //std::cout << "result: " << result << "\n";
  /*
  int classList[10]={0,0,0,0,0,0,0,0,0,0};

  for(auto obj : objectList){
    if (obj.classId <10){
    classList[obj.classId]++;
    }
    if (obj.classId == 1){
      std::cout << "obj.classId: " << obj.classId << std::endl;
      std::cout << "obj.detectionConfidence: " << obj.detectionConfidence << std::endl;
      std::cout << "obj.left: " << obj.left << std::endl;
      std::cout << "obj.top: " << obj.top << std::endl;
      std::cout << "obj.width: " << obj.width << std::endl;
      std::cout << "obj.height: " << obj.height << std::endl;
    }
  }
  for(int i=0;i<10;i++){
    std::cout << i << ":" << classList[i] << " ";
  }
  std::cout << std::endl;
  */
  return result;
}



CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);


