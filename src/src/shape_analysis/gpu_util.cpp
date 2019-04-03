/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/gpu_util.h>

namespace au_vision {

// 3.0 Spec is 16 blocks per SM. No API function for this, so we'll pick the
// lowest possible
int global_specBlocksPerSm = 16;

// Globals used internally by kernels
int global_threadsPerBlock = -1;
int global_totalSmBlocks = -1;

void initCudaAndOpenCv() {
  if (global_threadsPerBlock != -1) {
    ROS_INFO("Skipping CUDA / OpenCV initialization: already done");
  }

  // Initialize CUDA
  int devicesCount = 0;
  bool cudaSuccess = false;
  cudaDeviceProp deviceProperties;
  gpuErrorCheck(cudaGetDeviceCount(&devicesCount));
  for (int i = 0; i < devicesCount; ++i) {
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProperties, i));
    if (deviceProperties.major >= 3 && deviceProperties.minor >= 0) {
      // Note that CUDA orders GPUs with 0 being the fastest (so we will pick
      // the fastest working device)
      // Also does initialization internal to CUDA when we set the device
      gpuErrorCheck(cudaSetDevice(i));
      ROS_INFO("Using GPU with device number %i: %s", i, deviceProperties.name);

      // Calculate optimal block and thread counts for kernels
      global_totalSmBlocks =
          deviceProperties.multiProcessorCount * global_specBlocksPerSm;
      global_threadsPerBlock =
          deviceProperties.maxThreadsPerMultiProcessor / global_specBlocksPerSm;
      cudaSuccess = true;
      break;
    }
  }

  if (!cudaSuccess) {
    ROS_FATAL("Could not find a CUDA capable GPU with at least compute 3.0");
    ROS_BREAK();
  }

  // Check that OpenCV has CUDA support
  if (!cv::cuda::getCudaEnabledDeviceCount()) {
    ROS_FATAL(
        "OpenCV does not have CUDA support. Did you compile with it? Here's "
        "the build info for your OpenCV version:");
    std::string openCvBuildInfo = cv::getBuildInformation();
    ROS_INFO("%s", openCvBuildInfo.c_str());
    ROS_BREAK();
  }
}

}  // namespace au_vision
