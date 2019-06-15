#ifdef GPU

#ifndef AU_VISION_SHAPE_ANALYSIS_KERNELS_H
#define AU_VISION_SHAPE_ANALYSIS_KERNELS_H

#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis.h>

#include <ros/ros.h>

#include <builtin_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace au_vision {

// compresses every 8 bytes of a binary mask into a single byte
void callRasterToCompressed_device(unsigned char* binaryMask,
                                   uint8_t* compressed, int binaryMaskBytes);

// unpacks multiple images (into the compressed form)
void callPackedToCompressed_device(PixelPack* packed, uint32_t* compressed,
                                   int noImages, int compressedBytes,
                                   int imagePacks, cudaStream_t stream);

// computes bitwise XOR on corresponding elements of images
void callCompressedXOR_device(uint32_t* lhs, uint32_t* rhs, uint32_t* out,
                              int size, cudaStream_t stream);

// converts values into the number of bits set (population count)
void callPopulationCount_device(uint32_t* in, uint32_t* out, int size,
                                cudaStream_t stream);

// sums image
void callSumImage_device(uint32_t* image, int* out, int size,
                         cudaStream_t stream);

// Returns the area of each square in the grid segmented mask
void callBinRasterToGrid_device(uint8_t* shapeGrayMask, int squareRows,
                                int squareCols, int squareSize, int* bins);

// Returns the difference between two arrays of integers
void callRoughShapeAreaDifferences_device(int* binsRt, int* binsDb,
                                          int squareRows, int squareCols,
                                          int shapeCount, int* areaDifference,
                                          cudaStream_t stream);

// Returns the L1 color distances of the CIELAB AB channels from two arrays of
// unsigned chars Note: assumes rt_colors has length 3 (only a single rt shape)
void callColorDifferences_device(unsigned char* rt_colors,
                                 unsigned char* db_colors, int* differences,
                                 int shapeCount, int blackMargin,
                                 int whiteMargin, cudaStream_t stream);

}  // namespace au_vision

#endif

#endif
