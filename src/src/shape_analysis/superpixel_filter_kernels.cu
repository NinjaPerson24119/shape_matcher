#include <au_vision/shape_analysis/superpixel_filter_kernels.h>

namespace au_vision {

// Forward declarations
__global__ void drawAverageColors_device(gSLICr::Vector4i* bins,
                                         unsigned char* imageOut, size_t rows,
                                         size_t cols,
                                         cudaTextureObject_t grayImage);

// Thin wrappers

void callDrawAverageColors_device(int blocks, int threadsPerBlock,
                                  gSLICr::Vector4i* bins,
                                  unsigned char* imageOut, size_t rows,
                                  size_t cols, cudaTextureObject_t grayImage) {
  drawAverageColors_device<<<blocks, threadsPerBlock>>>(bins, imageOut, rows,
                                                        cols, grayImage);
  cudaDeviceSynchronize();            // Block until kernel queue finishes
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK
}

__global__ void drawAverageColors_device(gSLICr::Vector4i* bins,
                                         unsigned char* imageOut, size_t rows,
                                         size_t cols,
                                         cudaTextureObject_t grayImage) {
  // Check that thread idx is actually in range (otherwise skip)
  size_t sizeImage = rows * cols;
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sizeImage;
       idx += blockDim.x * gridDim.x) {
    // Fetch gray value (which cluster)
    int gray = tex1Dfetch<int>(grayImage, idx);

    // Draw pixel
    size_t y = idx / cols;
    size_t x = idx % cols;
    size_t loc = (3 * cols * y) + (3 * x);
    imageOut[loc] = bins[gray].x;
    imageOut[loc + 1] = bins[gray].y;
    imageOut[loc + 2] = bins[gray].z;
  }
}

}  // namespace au_vision
