#ifdef GPU

#ifndef AU_VISION_SUPERPIXEL_FILTER_KERNELS_H
#define AU_VISION_SUPERPIXEL_FILTER_KERNELS_H

#include <gSLICr_Lib/gSLICr.h>
#include <ros/ros.h>

#include <au_vision/shape_analysis/gpu_util.h>

namespace au_vision {

// Thin wrappers so that we can use CUDA C from C++

// Using the gray image, each pixel is colored by the indexed bin
extern "C" void callDrawAverageColors_device(int blocks, int threadsPerBlock,
                                             gSLICr::Vector4i* bins,
                                             unsigned char* imageOut,
                                             size_t rows, size_t cols,
                                             cudaTextureObject_t grayImage);

}  // namespace au_vision

#endif

#endif
