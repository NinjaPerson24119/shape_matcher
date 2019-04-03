/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/gpu_util_kernels.h>

namespace au_vision
{

//Forward declarations
__global__ void inRange_device(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSzb dst, int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2);
__global__ void simpleEdgeDetect_device(unsigned short* grayMask, unsigned char* binaryMask, int rows, int cols);
__device__ bool sameColor_device(unsigned short* grayMask, unsigned char* binaryMask, int rows, int cols, int idx, int movX, int movY);

//Kernel credits for inRange: https://github.com/opencv/opencv/issues/6295
void callInRange_device(const cv::cuda::GpuMat &src, const cv::Scalar &lowerb, const cv::Scalar &upperb,
                 cv::cuda::GpuMat &dst) {
   //Max block size of 1024 (as per spec)
   int m = global_threadsPerBlock;
   if(m > 32)
   {
     m = 32;
   }

  //Initialize the output (this was fixed from the sourced page)
  dst.create(src.rows, src.cols, CV_8UC1);

  int numRows = src.rows, numCols = src.cols;
  if (numRows == 0 || numCols == 0) return;
  // Attention! Cols Vs. Rows are reversed
  const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
  const dim3 blockSize(m, m, 1);

  inRange_device<<<gridSize, blockSize>>>(src, dst, lowerb[0], upperb[0], lowerb[1], upperb[1],
                                          lowerb[2], upperb[2]);
  cudaDeviceSynchronize();

  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callSimpleEdgeDetect_device(unsigned short* grayMask, unsigned char* binaryMask, int rows, int cols)
{
    //Each thread will sum a grid square
    int blocks = std::ceil((double)(rows * cols) / 32);
    int threadsPerBlock = 32;
  
    simpleEdgeDetect_device<<<blocks, threadsPerBlock>>>(grayMask, binaryMask, rows, cols);
    cudaDeviceSynchronize();  // Block until kernel queue finishes
    gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

__global__ void inRange_device(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSzb dst, int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= src.cols || y >= src.rows) return;

  uchar3 v = src(y, x);
  if (v.x >= lbc0 && v.x <= ubc0 && v.y >= lbc1 && v.y <= ubc1 && v.z >= lbc2 && v.z <= ubc2)
    dst(y, x) = 255;
  else
    dst(y, x) = 0;
}

__global__ void simpleEdgeDetect_device(unsigned short* grayMask, unsigned char* binaryMask, int rows, int cols)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < rows * cols)
  {
    bool isWhite = false;
    
    //If the color is not surrounded by the same color , set it to white 
    //(left, right, up, down)
    
    if(!sameColor_device(grayMask, binaryMask, rows, cols, idx, -1, 0))
    {
      isWhite = true;
    }
    if(!sameColor_device(grayMask, binaryMask, rows, cols, idx, 1, 0))
    {
      isWhite = true;
    }
    if(!sameColor_device(grayMask, binaryMask, rows, cols, idx, 0, -1))
    {
      isWhite = true;
    }
    if(!sameColor_device(grayMask, binaryMask, rows, cols, idx, 0, 1))
    {
      isWhite = true;
    }

    if(isWhite)
    {
      binaryMask[idx] = 255;
    }
    else {
      binaryMask[idx] = 0;
    }
  }
}

__device__ bool sameColor_device(unsigned short* grayMask, unsigned char* binaryMask, int rows, int cols, int idx, int movX, int movY)
{
  //Check if the adjacent location is in range
  int newIdx = idx + movX + cols * movY;
  if(newIdx >= 0 && newIdx < rows * cols)
  { 
      if(grayMask[idx] != grayMask[newIdx])
      {
        return false;
      }
      else {
        return true;
      }
  }
  else {
    return true;
  }
}


}
