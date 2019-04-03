/**
 * @author Nicholas Wengel
 */ 

#ifdef GPU

#ifndef AU_VISION_GPU_UTIL_H
#define AU_VISION_GPU_UTIL_H

#include <string>

// Now CUDA will be included  when gpu_util is included
#include <au_vision/shape_analysis/gpu_util_kernels.h>

namespace au_vision {

// The blocks per GPU multistream processor. Since there is no API function to
// get this value, we set it based on the minimum supported CUDA version spec
extern int global_specBlocksPerSm;
extern int global_threadsPerBlock;
extern int global_totalSmBlocks;

// Checks if OpenCV and CUDA are working, selects the fastest GPU, and returns
// the optimal kernel blocks + threadsPerBlock  Asserts if either CUDA is
// unavailable or OpenCV is not compiled with GPU support
void initCudaAndOpenCv();

}  // namespace au_vision

#endif

#endif
