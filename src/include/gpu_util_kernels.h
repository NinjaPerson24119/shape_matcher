#ifdef GPU

#ifndef AU_VISION_GPU_UTIL_KERNELS_H
#define AU_VISION_GPU_UTIL_KERNELS_H

#include <ros/ros.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glcorearb.h>
#include <cuda_gl_interop.h>

#include <builtin_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <au_vision/shape_analysis/gpu_util.h>

namespace au_vision {

// More elegant way to process and print CUDA errors
#define gpuErrorCheck(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool shouldAbort = true) {
  if (code != cudaSuccess) {
    ROS_FATAL("GPU Assert: %s, %s, %i", cudaGetErrorString(code), file, line);
    if (shouldAbort) {
      ROS_BREAK();
    }
  }
}

// Since OpenCV does not presently have a GPU version of cv::inrange, we'll roll
// our own Kernel. credits for inRange:
// https://github.com/opencv/opencv/issues/6295
extern "C" void callInRange_device(const cv::cuda::GpuMat &src,
                                   const cv::Scalar &lowerb,
                                   const cv::Scalar &upperb,
                                   cv::cuda::GpuMat &dst, cudaStream_t stream);

// Sets a binary mask outside the edges of spixels
extern "C" void callSimpleEdgeDetect_device(unsigned short *grayMask,
                                            unsigned char *binaryMask, int rows,
                                            int cols);

// Builds the autofilter's mask
extern "C" void callBuildMask_device(unsigned short *dstMask,
                                     unsigned short *valueMap,
                                     unsigned char *imageLab, int size);

}  // namespace au_vision

#endif

#endif
