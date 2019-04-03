/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <cmath>

namespace au_vision
{

//Forward declarations
__global__ void binRasterToGrid_device(unsigned char* shapeGrayMask, int squareRows, int squareCols, int squareSize, int* bins);
__global__ void roughShapeAreaDifferences_device(int* binsRt, int* binsDb, int squareRows, int squareCols, int shapeCount, int* areaDifference, unsigned char* colorsRt, unsigned char* colorsDb, int* colorDifference);

void callBinRasterToGrid_device(unsigned char* shapeGrayMask, int squareRows, int squareCols, int squareSize, int* bins)
{
  //Each thread will sum a grid square
  int blocks = std::ceil((double)(squareRows * squareCols) / 32);
  int threadsPerBlock = 32;

  binRasterToGrid_device<<<blocks, threadsPerBlock>>>(shapeGrayMask, squareRows, squareCols, squareSize, bins);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

void callRoughShapeAreaDifferences_device(int* binsRt, int* binsDb, int squareRows, int squareCols, int shapeCount, int* areaDifference, unsigned char* colorsRt, unsigned char* colorsDb, int* colorDifference)
{
  //Each thread will sum a grid square
  int blocks = std::ceil((double)(shapeCount) / 32);
  int threadsPerBlock = 32;
  
  roughShapeAreaDifferences_device<<<blocks, threadsPerBlock>>>(binsRt, binsDb, squareRows, squareCols, shapeCount, areaDifference, colorsRt, colorsDb, colorDifference);
  cudaDeviceSynchronize();  // Block until kernel queue finishes
  gpuErrorCheck( cudaGetLastError() ); //Verify that all went OK
}

__global__ void binRasterToGrid_device(unsigned char* shapeGrayMask, int squareRows, int squareCols, int squareSize, int* bins)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < squareRows * squareCols)
  {
    int col = idx % squareCols; 
    int row = idx / squareCols;

    //Sum the used area in the shape grid square
    bins[idx] = 0;
    for(int y = 0; y < squareSize; ++y)
    {
      for(int x = 0; x < squareSize; ++x)
      {
        size_t whichPixel = row * squareSize * squareCols * squareSize + col * squareSize;
        whichPixel += y * squareSize * squareCols + x;
      
        if(shapeGrayMask[whichPixel])
        {
          ++bins[idx];
        }
      }
    }
  }
}

__global__ void roughShapeAreaDifferences_device(int* binsRt, int* binsDb, int squareRows, int squareCols, int shapeCount, int* areaDifference, unsigned char* colorsRt, unsigned char* colorsDb, int* colorDifference)
{
  //It is assumed that binsRt is a single shape
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int area = squareRows * squareCols;
  if(idx < shapeCount)
  {
    areaDifference[idx] = 0;
    for(int i = 0; i < area; ++i)
    {
      //Accumulate difference
      //printf("idx: %i, i: %i, Subbing: %i - %i = %i\n", idx, i, binsRt[i], binsDb[idx*area+i], std::abs(binsRt[i] - binsDb[idx * area + i]));
      areaDifference[idx] += std::abs(binsRt[i] - binsDb[idx * area + i]);
    }
    //printf("KERNEL AREA DIFF: %i\n", areaDifference[idx]);
    colorDifference[idx] = 0;
    for(int i = 0; i < 3; ++i)
    {
      size_t colorLoc = idx * 3 + i;
      colorDifference[idx] += std::abs(colorsRt[colorLoc] - colorsDb[colorLoc]);
    }
  }
}

}
