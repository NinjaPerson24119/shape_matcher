/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/superpixel_filter_kernels.h>

namespace au_vision
{

//Forward declarations
__global__ void sumColorsToBins_device(gSLICr::Vector4i* bins, size_t sizeImage, cudaTextureObject_t colorImage, cudaTextureObject_t grayImage);
__global__ void divideColorsInBins_device(gSLICr::Vector4i* bins, size_t binsCount);
__global__ void drawAverageColors_device(gSLICr::Vector4i* bins, unsigned char* imageOut, size_t rows, size_t cols, cudaTextureObject_t grayImage);
__global__ void convertImageVectorsToOpenCv_device(const gSLICr::Vector4u* imageIn, unsigned char* imageOut, size_t rows, size_t cols);
__global__ void convertImageOpenCvToVectors_device(const unsigned char* imageIn, gSLICr::Vector4u* imageOut, size_t rows, size_t cols);

//Thin wrappers
void callSumColorsToBins_device(int blocks, int threadsPerBlock, gSLICr::Vector4i* bins, size_t sizeImage, cudaTextureObject_t colorImage, cudaTextureObject_t grayImage)
{
  sumColorsToBins_device<<<blocks, threadsPerBlock>>>(bins, sizeImage, colorImage, grayImage);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callDivideColorsInBins_device(int blocks, int threadsPerBlock, gSLICr::Vector4i* bins, size_t binsCount)
{
  divideColorsInBins_device<<<blocks, threadsPerBlock>>>(bins, binsCount);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callDrawAverageColors_device(int blocks, int threadsPerBlock, gSLICr::Vector4i* bins, unsigned char* imageOut, size_t rows, size_t cols, cudaTextureObject_t grayImage)
{
  drawAverageColors_device<<<blocks, threadsPerBlock>>>(bins, imageOut, rows, cols, grayImage);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callConvertImageVectorsToOpenCv_device(int blocks, int threadsPerBlock, const gSLICr::Vector4u* imageIn, unsigned char* imageOut, size_t rows, size_t cols)
{
  convertImageVectorsToOpenCv_device<<<blocks, threadsPerBlock>>>(imageIn, imageOut, rows, cols);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callConvertImageOpenCvToVectors_device(int blocks, int threadsPerBlock, const unsigned char* imageIn, gSLICr::Vector4u* imageOut, size_t rows, size_t cols)
{
  convertImageOpenCvToVectors_device<<<blocks, threadsPerBlock>>>(imageIn, imageOut, rows, cols);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

//Sums the colors in an image into bins, based on a greyscale image
//Note that texture references are used, hence there are no image parameters
//bins is the array of bins to sum to
//sizeImage is the 1D length of the image
__global__ void sumColorsToBins_device(gSLICr::Vector4i* bins, size_t sizeImage, cudaTextureObject_t colorImage, cudaTextureObject_t grayImage)
{
  //Check that thread idx is actually in range (otherwise skip)
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sizeImage; idx += blockDim.x * gridDim.x)
  {
    //Fetch from texture memory; hopefully the target is cached
    uchar4 color = tex1Dfetch<uchar4>(colorImage, idx);
    int gray = tex1Dfetch<int>(grayImage, idx);

    //Prepare for a large amount of serialization!
    //These operations are atomic since multiple threads can try to add to the same bin at the same time
    atomicAdd(&bins[gray].x, static_cast<int>(color.x));
    atomicAdd(&bins[gray].y, static_cast<int>(color.y));
    atomicAdd(&bins[gray].z, static_cast<int>(color.z));
    atomicAdd(&bins[gray].w, 1);
  }
}

//Divides each color by its w component
//bins is the array of bins to sum to
//binsCount is the number of bins
__global__ void divideColorsInBins_device(gSLICr::Vector4i* bins, size_t binsCount)
{
  //Check that thread idx is actually in range (otherwise skip)
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < binsCount; idx += blockDim.x * gridDim.x)
  {
    //Perform division
    bins[idx].x /= bins[idx].w;
    bins[idx].y /= bins[idx].w;
    bins[idx].z /= bins[idx].w;
  }
}

__global__ void drawAverageColors_device(gSLICr::Vector4i* bins, unsigned char* imageOut, size_t rows, size_t cols, cudaTextureObject_t grayImage)
{
  //Check that thread idx is actually in range (otherwise skip)
  size_t sizeImage = rows * cols;
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sizeImage; idx += blockDim.x * gridDim.x)
  {
    //Fetch gray value (which cluster)
    int gray = tex1Dfetch<int>(grayImage, idx);

    //Draw pixel
    size_t y = idx / cols;
    size_t x = idx % cols;
    size_t loc = (3 * cols * y) + (3 * x);
    imageOut[loc] = bins[gray].z;
    imageOut[loc + 1] = bins[gray].y;
    imageOut[loc + 2] = bins[gray].x;
  }
}

__global__ void convertImageVectorsToOpenCv_device(const gSLICr::Vector4u* imageIn, unsigned char* imageOut, size_t rows, size_t cols)
{
  //Check that thread idx is actually in range (otherwise skip)
  size_t sizeImage = rows * cols;
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sizeImage; idx += blockDim.x * gridDim.x)
  {
    //Assign pixels (3 channels is assumed)
    //Note that b,g,r notation is arbitrary and of no consequence if consistent
    size_t y = idx / cols;
    size_t x = idx % cols;
    size_t loc = (3 * cols * y) + (3 * x);
    imageOut[loc] = imageIn[idx].b;
    imageOut[loc + 1] = imageIn[idx].g;
    imageOut[loc + 2] = imageIn[idx].r;
  }
}

__global__ void convertImageOpenCvToVectors_device(const unsigned char* imageIn, gSLICr::Vector4u* imageOut, size_t rows, size_t cols)
{
  //Check that thread idx is actually in range (otherwise skip)
  size_t sizeImage = rows * cols;
  for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < sizeImage; idx += blockDim.x * gridDim.x)
  {
    //Assign pixels (3 channels is assumed)
    //Note that b,g,r notation is arbitrary and of no consequence if consistent
    size_t y = idx / cols;
    size_t x = idx % cols;
    size_t loc = (3 * cols * y) + (3 * x);
    imageOut[idx].b = imageIn[loc];
    imageOut[idx].g = imageIn[loc + 1];
    imageOut[idx].r = imageIn[loc + 2];
  }
}

}
