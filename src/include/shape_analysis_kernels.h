/**
 * @author Nicholas Wengel
 */ 

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

#include <vector>

namespace au_vision {

// Returns the area of each square in the grid segmented mask
void callBinRasterToGrid_device(unsigned char* shapeGrayMask, int squareRows,
                                int squareCols, int squareSize, int* bins);

// Returns the difference between two arrays of integers
void callRoughShapeAreaDifferences_device(int* binsRt, int* binsDb,
                                          int squareRows, int squareCols,
                                          int shapeCount, int* areaDifference,
                                          unsigned char* colorsRt,
                                          unsigned char* colorsDb,
                                          int* colorDifference);

}  // namespace au_vision

#endif

#endif
