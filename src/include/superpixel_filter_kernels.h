/**
 * @author Nicholas Wengel
 */ 

#ifdef GPU

#ifndef AU_VISION_SUPERPIXEL_FILTER_KERNELS_H
#define AU_VISION_SUPERPIXEL_FILTER_KERNELS_H

#include <gSLICr_Lib/gSLICr.h>
#include <ros/ros.h>

#include <au_vision/shape_analysis/gpu_util.h>

namespace au_vision {

// Thin wrappers so that we can use CUDA C from C++

// Adds each pixel in a color image to a bin, where the bin is the corresponding
// value in the gray image  The w component of each bin will contain the number
// of pixels added
extern "C" void callSumColorsToBins_device(int blocks, int threads,
                                           gSLICr::Vector4i* bins,
                                           size_t sizeImage,
                                           cudaTextureObject_t colorImage,
                                           cudaTextureObject_t grayImage);

// Divides each bin by its w component
extern "C" void callDivideColorsInBins_device(int blocks, int threads,
                                              gSLICr::Vector4i* bins,
                                              size_t binsCount);

// Using the gray image, each pixel is colored by the indexed bin
extern "C" void callDrawAverageColors_device(int blocks, int threadsPerBlock,
                                             gSLICr::Vector4i* bins,
                                             unsigned char* imageOut,
                                             size_t rows, size_t cols,
                                             cudaTextureObject_t grayImage);

// Converts gSLICr vector array to an OpenCV array
extern "C" void callConvertImageVectorsToOpenCv_device(
    int blocks, int threadsPerBlock, const gSLICr::Vector4u* imageIn,
    unsigned char* imageOut, size_t rows, size_t cols);

// Converts OpenCV array to a gSLICr vector array
extern "C" void callConvertImageOpenCvToVectors_device(
    int blocks, int threadsPerBlock, const unsigned char* imageIn,
    gSLICr::Vector4u* imageOut, size_t rows, size_t cols);
}  // namespace au_vision

#endif

#endif
