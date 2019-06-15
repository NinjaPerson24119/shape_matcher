#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <cassert>
#include <cmath>

namespace au_vision {

// Forward declarations
__global__ void rasterToCompressed_device(unsigned char* binaryMask,
                                          uint8_t* compressed,
                                          int binaryMaskBytes);

__global__ void packedToCompressed_device(PixelPack* packed,
                                          uint32_t* compressed, int noImages,
                                          int compressedBytes, int imagePacks);

__global__ void compressedXOR_device(uint32_t* lhs, uint32_t* rhs,
                                     uint32_t* out, int size);

__global__ void populationCount_device(uint32_t* in, uint32_t* out, int size);

__global__ void sumImage_device(uint32_t* image, int* out, int size);

__global__ void colorDifferences_device(unsigned char* rtColors,
                                        unsigned char* dbColors,
                                        int* differences, int shapeCount,
                                        int blackMargin, int whiteMargin);

__device__ bool isBlack(unsigned char* color, int blackMargin);
__device__ bool isWhite(unsigned char* color, int whiteMargin);

__global__ void binRasterToGrid_device(uint8_t* shapeGrayMask, int squareRows,
                                       int squareCols, int squareSize,
                                       int* bins);

__global__ void roughShapeAreaDifferences_device(int* binsRt, int* binsDb,
                                                 int squareRows, int squareCols,
                                                 int shapeCount,
                                                 int* areaDifference);

void callRasterToCompressed_device(unsigned char* binaryMask,
                                   uint8_t* compressed, int binaryMaskBytes) {
  assert(binaryMaskBytes % 8 == 0);

  // Divide image into block of 32 threads, each thread compressing 8 bytes
  int blocks = std::ceil((double)(binaryMaskBytes / (32 * 8)));
  int threadsPerBlock = 32;

  rasterToCompressed_device<<<blocks, threadsPerBlock>>>(binaryMask, compressed,
                                                         binaryMaskBytes);

  cudaDeviceSynchronize();            // Block until kernel queue finishes
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK
}

void callPackedToCompressed_device(PixelPack* packed, uint32_t* compressed,
                                   int noImages, int compressedBytes,
                                   int imagePacks, cudaStream_t stream) {
  // note that each thread will unpack a single image
  int blocks = std::ceil((double)noImages / 32.0);
  int threadsPerBlock = 32;

  packedToCompressed_device<<<blocks, threadsPerBlock, 0, stream>>>(
      packed, compressed, noImages, compressedBytes, imagePacks);
}

void callCompressedXOR_device(uint32_t* lhs, uint32_t* rhs, uint32_t* out,
                              int size, cudaStream_t stream) {
  int blocks = std::ceil((double)size / 32.0);
  int threadsPerBlock = 32;

  compressedXOR_device<<<blocks, threadsPerBlock, 0, stream>>>(lhs, rhs, out,
                                                               size);
}

void callPopulationCount_device(uint32_t* in, uint32_t* out, int size,
                                cudaStream_t stream) {
  int blocks = std::ceil((double)size / 32.0);
  int threadsPerBlock = 32;

  populationCount_device<<<blocks, threadsPerBlock, 0, stream>>>(in, out, size);
}

void callSumImage_device(uint32_t* image, int* out, int size,
                         cudaStream_t stream) {
  // note that each thread will sum a single image
  int blocks = std::ceil((double)size / 32.0);
  int threadsPerBlock = 32;

  sumImage_device<<<blocks, threadsPerBlock, 0, stream>>>(image, out, size);
}

void callColorDifferences_device(unsigned char* rtColors,
                                 unsigned char* dbColors, int* differences,
                                 int shapeCount, int blackMargin,
                                 int whiteMargin, cudaStream_t stream) {
  // Each thread will calculate a single rt vs. db difference
  int blocks = std::ceil((double)shapeCount / 32.0);
  int threadsPerBlock = 32;

  colorDifferences_device<<<blocks, threadsPerBlock, 0, stream>>>(
      rtColors, dbColors, differences, shapeCount, blackMargin, whiteMargin);
}

void callBinRasterToGrid_device(uint8_t* shapeGrayMask, int squareRows,
                                int squareCols, int squareSize, int* bins) {
  // Each thread will sum a grid square
  int blocks = std::ceil((double)(squareRows * squareCols) / 32);
  int threadsPerBlock = 32;

  binRasterToGrid_device<<<blocks, threadsPerBlock>>>(
      shapeGrayMask, squareRows, squareCols, squareSize, bins);
  cudaDeviceSynchronize();            // Block until kernel queue finishes
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK
}

void callRoughShapeAreaDifferences_device(int* binsRt, int* binsDb,
                                          int squareRows, int squareCols,
                                          int shapeCount, int* areaDifference,
                                          cudaStream_t stream) {
  // Each thread will sum a grid square
  int blocks = std::ceil((double)(shapeCount) / 32);
  int threadsPerBlock = 32;

  roughShapeAreaDifferences_device<<<blocks, threadsPerBlock, 0, stream>>>(
      binsRt, binsDb, squareRows, squareCols, shapeCount, areaDifference);
}

__global__ void rasterToCompressed_device(unsigned char* binaryMask,
                                          uint8_t* compressed,
                                          int binaryMaskBytes) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int maskLoc = idx * 8;
  if (maskLoc < binaryMaskBytes) {
    compressed[idx] = 0;
    // note: this assumes that machine endianness is consistent between DB
    // creator and matcher
    for (int i = 0; i < 8; ++i) {
      // check that pixel is set
      if (binaryMask[maskLoc + i]) {
        // set corresponding bit
        compressed[idx] |= (unsigned char)1 << i;
      }
    }
  }
}

__global__ void packedToCompressed_device(PixelPack* packed,
                                          uint32_t* compressed, int noImages,
                                          int compressedBytes, int imagePacks) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < noImages) {
    PixelPack* pack = packed + imagePacks * idx;
    // fill as uchars
    uint8_t* out = (uint8_t*)(compressed) + compressedBytes * idx;
    while (pack->repetitions != 0) {
      for (int i = 0; i < pack->repetitions; ++i) {
        out[i] = pack->type;
      }
      out += pack->repetitions;
      pack += sizeof(PixelPack);
    }
  }
}

__global__ void compressedXOR_device(uint32_t* lhs, uint32_t* rhs,
                                     uint32_t* out, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    out[idx] = lhs[idx] ^ rhs[idx];
  }
}

__global__ void populationCount_device(uint32_t* in, uint32_t* out, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    out[idx] = __popc(in[idx]);
  }
}

__global__ void sumImage_device(uint32_t* image, int* out, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    atomicAdd(out, image[idx]);
  }
}

__global__ void colorDifferences_device(unsigned char* rtColors,
                                        unsigned char* dbColors,
                                        int* differences, int shapeCount,
                                        int blackMargin, int whiteMargin) {
  // It is assumed that binsRt is a single shape
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < shapeCount) {
    differences[idx] = 0;
    // note: start at i=1 to skip L channel of CIELAB
    size_t colorLoc = idx * 3;

    // check general case
    for (int i = 1; i < 3; ++i) {
      differences[idx] += std::abs(rtColors[i] - dbColors[colorLoc + i]);
    }

    // check special black and white cases
    int lhs = 0, rhs = 0;
    lhs = isBlack(rtColors, blackMargin) ? 1 : lhs;
    lhs = isWhite(rtColors, whiteMargin) ? 2 : lhs;
    rhs = isBlack(&dbColors[colorLoc], blackMargin) ? 1 : rhs;
    rhs = isWhite(&dbColors[colorLoc], whiteMargin) ? 2 : rhs;
    if (lhs || rhs) {
      if (lhs == rhs) {
        differences[idx] = 0;
      } else {
        differences[idx] = 510;
      }
    }
  }
}

__device__ bool isBlack(unsigned char* color, int blackMargin) {
  bool centralA = std::abs((int)color[1] - 128) <= blackMargin;
  bool centralB = std::abs((int)color[2] - 128) <= blackMargin;
  bool edge = color[0] <= blackMargin * 2;
  if (centralA && centralB && edge) {
    return true;
  } else {
    return false;
  }
}

__device__ bool isWhite(unsigned char* color, int whiteMargin) {
  bool centralA = std::abs((int)color[1] - 128) <= whiteMargin;
  bool centralB = std::abs((int)color[2] - 128) <= whiteMargin;
  bool edge = color[0] >= 255 - whiteMargin * 2;
  if (centralA && centralB && edge) {
    return true;
  } else {
    return false;
  }
}

__global__ void binRasterToGrid_device(uint8_t* shapeGrayMask, int squareRows,
                                       int squareCols, int squareSize,
                                       int* bins) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < squareRows * squareCols) {
    int col = idx % squareCols;
    int row = idx / squareCols;

    // Sum the used area in the shape grid square
    bins[idx] = 0;
    for (int y = 0; y < squareSize; ++y) {
      for (int x = 0; x < squareSize; ++x) {
        size_t whichPixel =
            row * squareSize * squareCols * squareSize + col * squareSize;
        whichPixel += y * squareSize * squareCols + x;

        if (shapeGrayMask[whichPixel]) {
          ++bins[idx];
        }
      }
    }
  }
}

__global__ void roughShapeAreaDifferences_device(int* binsRt, int* binsDb,
                                                 int squareRows, int squareCols,
                                                 int shapeCount,
                                                 int* areaDifference) {
  // It is assumed that binsRt is a single shape
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int area = squareRows * squareCols;
  if (idx < shapeCount) {
    areaDifference[idx] = 0;
    for (int i = 0; i < area; ++i) {
      areaDifference[idx] += std::abs(binsRt[i] - binsDb[idx * area + i]);
    }
  }
}

}  // namespace au_vision
