#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <au_vision/shape_analysis/shape_analysis_matcher.h>
#include <ros/ros.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <list>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

namespace au_vision {

const char* dbHeaderString = "SHAPE_ANALYSIS_1";

void packedToCompressed(const std::vector<PixelPack>& packedImage,
                        uint8_t* compressed) {
  int i = 0;
  for (int k = 0; k < packedImage.size(); ++k) {
    for (int l = 0; l < packedImage[k].repetitions; ++l) {
      compressed[i + l] = packedImage[k].type;
    }
    i += packedImage[k].repetitions;
  }
}

std::vector<PixelPack> compressedToPacked(const uint8_t* compressed,
                                          int sizeBytes, int imageWidth) {
  // Pack compressed image
  std::vector<PixelPack> packs;
  int i = 1;  // iterate image bytes
  ROS_ASSERT(sizeBytes > 2);
  PixelPack p = {compressed[0], 1};
  while (i != sizeBytes) {
    // complete pack if:
    // - next byte doesn't equal the current type
    if (compressed[i] != p.type || i % imageWidth == 0) {
      packs.emplace_back(p);
      p.repetitions = 1;
      p.type = compressed[i];
    } else {
      ++p.repetitions;
    }

    // last byte condition
    if (i + 1 == sizeBytes) {
      packs.emplace_back(p);
    }

    // continue
    ++i;
  }
  for (auto i : packs) {
    ROS_ASSERT(i.repetitions);
  }
  return packs;
}

void averageContourColors(const cv::Mat& image,
                          const std::vector<std::vector<cv::Point>>& contours,
                          std::vector<ShapeColor>& colors, int filterSize,
                          cv::Mat* outSampleRegions) {
  ROS_ASSERT(image.type() == CV_8UC3);

  // stream to run kernels in parallel
  cv::cuda::Stream stream;

  // render
  std::vector<cv::cuda::GpuMat> dev_clip;
  std::vector<cv::Rect> bounds;
  for (int i = 0; i < contours.size(); ++i) {
    // render
    cv::Mat render(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> temp;
    temp.push_back(contours[i]);
    cv::drawContours(render, temp, -1, cv::Scalar(255), CV_FILLED);

    // get bounding box and adjust for dilation size
    bounds.emplace_back(cv::boundingRect(contours[i]));
    int adj = filterSize;
    bounds[i].x = bounds[i].x - adj > 0 ? bounds[i].x - adj : 0;
    bounds[i].y = bounds[i].y - adj > 0 ? bounds[i].y - adj : 0;
    bounds[i].width = bounds[i].x + adj < image.cols ? bounds[i].width + adj
                                                     : image.cols - bounds[i].x;
    bounds[i].height = bounds[i].y + adj < image.rows
                           ? bounds[i].height + adj
                           : image.rows - bounds[1].y;

    // load to GPU
    cv::Mat clip(render, bounds[i]);
    dev_clip.emplace_back(cv::cuda::GpuMat());
    dev_clip[i].upload(clip, stream);
  }
  stream.waitForCompletion();

  // synchronous GPU calls
  cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size(filterSize, filterSize));
  cv::Ptr<cv::cuda::Filter> efilter =
      cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, elem);
  cv::Ptr<cv::cuda::Filter> dfilter =
      cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, elem);

  // erode / dilate
  std::vector<cv::cuda::GpuMat> dev_eroded, dev_dilated;
  for (int i = 0; i < contours.size(); ++i) {
    dev_eroded.emplace_back(cv::cuda::GpuMat());
    dev_dilated.emplace_back(cv::cuda::GpuMat());
    efilter->apply(dev_clip[i], dev_eroded[i], stream);
    dfilter->apply(dev_clip[i], dev_dilated[i], stream);
  }
  stream.waitForCompletion();

  // get interior / exterior
  for (int i = 0; i < contours.size(); ++i) {
    cv::cuda::bitwise_xor(dev_clip[i], dev_eroded[i], dev_eroded[i],
                          cv::noArray(), stream);
    cv::cuda::bitwise_xor(dev_clip[i], dev_dilated[i], dev_dilated[i],
                          cv::noArray(), stream);
  }
  stream.waitForCompletion();

  // upload image clips to GPU
  std::vector<cv::cuda::GpuMat> dev_image_clips;
  for (int i = 0; i < contours.size(); ++i) {
    cv::Mat clip(image, bounds[i]);
    dev_image_clips.emplace_back(cv::cuda::GpuMat());
    dev_image_clips[i].upload(clip, stream);
  }
  stream.waitForCompletion();

  // sum elements where mask is set
  std::vector<cv::Scalar> intColor(contours.size());
  std::vector<cv::Scalar> extColor(contours.size());
  for (int i = 0; i < contours.size(); ++i) {
    intColor[i] = cv::cuda::sum(dev_image_clips[i], dev_eroded[i]);
    extColor[i] = cv::cuda::sum(dev_image_clips[i], dev_dilated[i]);
  }

  // count non-zero
  std::vector<cv::cuda::HostMem> intCount(contours.size());
  std::vector<cv::cuda::HostMem> extCount(contours.size());
  for (int i = 0; i < contours.size(); ++i) {
    cv::cuda::countNonZero(dev_eroded[i], intCount[i], stream);
    cv::cuda::countNonZero(dev_dilated[i], extCount[i], stream);
  }
  stream.waitForCompletion();

  // construct output
  colors.clear();
  for (int i = 0; i < contours.size(); ++i) {
    // fetch
    int ic;
    const cv::Mat ic_mat(1, 1, CV_32SC1, &ic);
    intCount[i].createMatHeader().copyTo(ic_mat);
    int ec;
    const cv::Mat ec_mat(1, 1, CV_32SC1, &ec);
    extCount[i].createMatHeader().copyTo(ec_mat);

    // divide
    intColor[i] /= ic;
    extColor[i] /= ec;

    // push
    ShapeColor c;
    c.interior = intColor[i];
    c.exterior = extColor[i];
    c.validInterior = true;
    c.validExterior = true;

    colors.emplace_back(c);
  }

  // debug output
  if (outSampleRegions != nullptr) {
    *outSampleRegions = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < contours.size(); ++i) {
      cv::Mat intMask, extMask;
      dev_eroded[i].download(intMask);
      dev_dilated[i].download(extMask);
      intMask *= 0.3;
      extMask *= 0.4;
      cv::Mat where(*outSampleRegions, bounds[i]);
      intMask.copyTo(where, intMask);
      extMask.copyTo(where, extMask);
    }
  }
}

ColorRange::ColorRange(const cv::Scalar& target, int variance) {
  lower = target;
  upper = target;
  for (int i = 0; i < 3; ++i) {
    ((int)lower[i] - variance > 0) ? lower[i] -= variance : lower[i] = 0;
    ((int)upper[i] + variance <= 255) ? upper[i] += variance : upper[i] = 255;
  }
}

bool ColorRange::contains(const ColorRange& other) {
  int nested = 0;
  for (int k = 0; k < 3; ++k) {
    if (lower[k] <= other.lower[k] && upper[k] >= other.upper[k]) {
      ++nested;
    }
  }
  if (nested == 3) {
    return true;
  } else {
    return false;
  }
}

void ShapeDb::addGroup(const std::vector<Shape>& newShapes,
                       const std::string& name, const cv::Scalar& pose,
                       const cv::Rect& groupBound) {
  if (!newShapes.size()) {
    return;
  }

  // check params have been set
  ROS_ASSERT(frameBufferWidth && frameBufferHeight);

  // Build group
  ShapeGroup group;
  group.pose = pose;
  for (int i = 0; i < newShapes.size(); ++i) {
    group.shapeIndices.push_back(i + shapes.size());
  }
  group.name = name;
  group.bound = groupBound;
  groups.emplace_back(group);

  // Iterate over shapes
  for (auto s : newShapes) {
    // Render the contour
    cv::Mat raster(frameBufferHeight, frameBufferWidth, CV_8UC1, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(s.contour);
    cv::drawContours(raster, contours, -1, cv::Scalar(255), CV_FILLED);

    // Compress every 8 uchars into a single byte of bits (with GPU)
    unsigned char* dev_raster;
    size_t rasterBytes = sizeof(unsigned char) * raster.rows * raster.cols;
    gpuErrorCheck(cudaMalloc(&dev_raster, rasterBytes));
    gpuErrorCheck(cudaMemcpy(dev_raster, raster.data, rasterBytes,
                             cudaMemcpyHostToDevice));

    size_t compressedBytes = rasterBytes / 8;
    uint8_t* compressed = new unsigned char[compressedBytes];
    uint8_t* dev_compressed;
    gpuErrorCheck(cudaMalloc(&dev_compressed, compressedBytes));

    callRasterToCompressed_device(dev_raster, dev_compressed, rasterBytes);

    gpuErrorCheck(cudaMemcpy(compressed, dev_compressed, compressedBytes,
                             cudaMemcpyDeviceToHost));

    // Pack compressed image
    packedImages.emplace_back(
        compressedToPacked(compressed, compressedBytes, frameBufferWidth / 8));

    // grids
    size_t gridArea = gridRows * gridCols;
    int* bins = new int[gridArea];
    int* dev_bins;
    gpuErrorCheck(cudaMalloc(&dev_bins, gridArea * sizeof(int)));

    // Call kernel
    callBinRasterToGrid_device(dev_raster, gridRows, gridCols, squareSize,
                               dev_bins);

    // Retrieve bins
    gpuErrorCheck(cudaMemcpy(bins, dev_bins, gridArea * sizeof(int),
                             cudaMemcpyDeviceToHost));

    // Append the bins to the existing grids
    grids.reserve(grids.size() + gridArea);
    std::copy(bins, bins + gridArea, std::back_inserter(grids));

    // Add color
    for (int i = 0; i < 3; ++i) {
      interiorColors.push_back(s.color.interior[i]);
      exteriorColors.push_back(s.color.exterior[i]);
    }

    // Deallocate
    gpuErrorCheck(cudaFree(dev_bins));
    gpuErrorCheck(cudaFree(dev_raster));
    gpuErrorCheck(cudaFree(dev_compressed));
    delete[] compressed;
    delete[] bins;

    shapes.push_back(s);
  }
}

void ShapeDb::loadToGpu() {
  ROS_ASSERT(!loadedGpu);
  ROS_ASSERT(shapes.size());

  // for async CUDA
  cudaStream_t stream;
  gpuErrorCheck(cudaStreamCreate(&stream));

  // allocate

  size_t gridsBytes = gridRows * gridCols * shapes.size() * sizeof(int);
  gpuErrorCheck(cudaMalloc(&dev_grids, gridsBytes));

  // get maximum number of packs per image
  // pixel packs on GPU will be uploaded in same size memory blocks for each
  // image regardless of their individual pixel packs (to avoid architectural
  // issues with device pointers to device pointers)
  maxPacks = 0;
  for (int i = 0; i < packedImages.size(); ++i) {
    if (packedImages[i].size() > maxPacks) {
      maxPacks = packedImages[i].size();
    }
  }
  ROS_ASSERT(maxPacks);
  gpuErrorCheck(cudaMalloc(&dev_packedImages,
                           packedImages.size() * sizeof(PixelPack) * maxPacks));

  size_t colorsBytes = shapes.size() * sizeof(unsigned char) * 3;
  gpuErrorCheck(cudaMalloc(&dev_interiorColors, colorsBytes));
  gpuErrorCheck(cudaMalloc(&dev_exteriorColors, colorsBytes));

  // upload to GPU
  gpuErrorCheck(cudaMemcpyAsync(dev_grids, &grids[0], gridsBytes,
                                cudaMemcpyHostToDevice, stream));

  for (int i = 0; i < packedImages.size(); ++i) {
    gpuErrorCheck(cudaMemcpyAsync(dev_packedImages + maxPacks * i,
                                  &packedImages[i][0],
                                  packedImages[i].size() * sizeof(PixelPack),
                                  cudaMemcpyHostToDevice, stream));
  }

  gpuErrorCheck(cudaMemcpyAsync(dev_interiorColors, &interiorColors[0],
                                colorsBytes, cudaMemcpyHostToDevice, stream));
  gpuErrorCheck(cudaMemcpyAsync(dev_exteriorColors, &exteriorColors[0],
                                colorsBytes, cudaMemcpyHostToDevice, stream));

  gpuErrorCheck(cudaStreamSynchronize(stream));
  gpuErrorCheck(cudaStreamDestroy(stream));

  loadedGpu = true;
}

void ShapeDb::unloadFromGpu() {
  ROS_ASSERT(loadedGpu);

  gpuErrorCheck(cudaFree(dev_grids));
  gpuErrorCheck(cudaFree(dev_packedImages));
  gpuErrorCheck(cudaFree(dev_interiorColors));
  gpuErrorCheck(cudaFree(dev_exteriorColors));

  dev_grids = nullptr;
  dev_packedImages = nullptr;
  dev_interiorColors = nullptr;
  dev_exteriorColors = nullptr;

  loadedGpu = false;
}

ShapeDb::~ShapeDb() {
  if (loadedGpu) {
    unloadFromGpu();
  }
}

void centerAndMaximizeContour(std::vector<cv::Point>& contour, int width,
                              int height) {
  // Find bounding box
  cv::Rect bound = cv::boundingRect(contour);

  // Resize bound while maintaining aspect ratio
  double aspectRatio = width / height;
  double aspectRatioRt = (double)bound.width / (double)bound.height;
  int maxedHeight = height;
  int maxedWidth = aspectRatioRt * maxedHeight;
  if (maxedWidth > width) {
    maxedWidth = width;
    maxedHeight = maxedWidth / aspectRatioRt;
  }

  // Center new bound
  cv::Rect newBound(width / 2 - maxedWidth / 2, height / 2 - maxedHeight / 2,
                    maxedWidth, maxedHeight);

  // Move points
  for (int i = 0; i < contour.size(); ++i) {
    contour[i].x =
        (int)(((double)(contour[i].x - bound.x) / (double)bound.width) *
                  (double)newBound.width +
              newBound.x);
    contour[i].y =
        (int)(((double)(contour[i].y - bound.y) / (double)bound.height) *
                  (double)newBound.height +
              newBound.y);
  }
}

void cullContours(std::vector<std::vector<cv::Point>>& contours, int imageWidth,
                  int imageHeight, double minArea, int minPoints,
                  int maxPoints) {
  // Calculate contour areas
  std::vector<double> contourAreas(contours.size());
  for (int i = 0; i < contours.size(); ++i) {
    contourAreas[i] = cv::contourArea(contours[i]);
  }

  // Remove noise contours
  {
    bool repeat;
    do {
      repeat = false;
      for (int i = 0; i < contours.size(); ++i) {
        if (contourAreas[i] < minArea) {
          contours.erase(contours.begin() + i);
          contourAreas.erase(contourAreas.begin() + i);
          repeat = true;
          break;
        }
      }
    } while (repeat);
  }

  // Remove contours touching the image edge (also takes care of screen border)
  {
    int i = 0;
    while (true && contours.size() != 0) {
      bool reset = false;
      for (int j = 0; j < contours[i].size(); ++j) {
        if (contours[i][j].x == 0 || contours[i][j].x == imageWidth - 1 ||
            contours[i][j].y == 0 || contours[i][j].y == imageHeight - 1) {
          contourAreas.erase(contourAreas.begin() + i);
          contours.erase(contours.begin() + i);
          i = 0;
          reset = true;
          break;
        }
      }
      if (reset == false) {
        ++i;
      }
      if (i == contours.size()) {
        break;
      }
    }
  }

  // Remove contours with too few points (require > 2)
  {
    int max = (maxPoints == 0) ? std::numeric_limits<int>::max() : maxPoints;
    int i = 0;
    while (contours.size() != 0) {
      if (contours[i].size() < 3 || contours[i].size() < minPoints ||
          contours[i].size() > max) {
        contourAreas.erase(contourAreas.begin() + i);
        contours.erase(contours.begin() + i);
        i = 0;
      }
      ++i;
      if (i == contours.size()) {
        break;
      }
    }
  }
}

void loadShapeAnalysisDatabase(const std::string& filename, ShapeDb& db) {
  // Open the file
  ROS_INFO("Reading shape analysis database at: %s", filename.c_str());
  std::ifstream file(filename);

  // Validate
  if (!file.good()) {
    ROS_FATAL("Could not open shape analysis database.");
    ROS_BREAK();
    return;
  }

  // Enable exceptions for easier error processing
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  // Read header
  std::string line;
  try {
    std::getline(file, line);
    if (line != dbHeaderString) {
      ROS_FATAL(
          "Could not load shape analysis database. Invalid header (maybe wrong "
          "DB version?).");
      ROS_BREAK();
    }
  } catch (...) {
    ROS_FATAL("Could not read header");
    ROS_BREAK();
  }

  // Read render parameters
  try {
    std::getline(file, line);
    db.gridRows = std::stoi(line);
    std::getline(file, line);
    db.gridCols = std::stoi(line);
    std::getline(file, line);
    db.squareSize = std::stoi(line);
    std::getline(file, line);
    db.frameBufferHeight = std::stoi(line);
    std::getline(file, line);
    db.frameBufferWidth = std::stoi(line);
    std::getline(file, line);
    db.renderDistance = std::stod(line);
  } catch (...) {
    ROS_FATAL("Could not read render params");
    ROS_BREAK();
  }

  // Read shape number
  int shapeNum;
  try {
    std::getline(file, line);
    shapeNum = std::stoi(line);
  } catch (...) {
    ROS_FATAL("Could not read shape count.");
    ROS_BREAK();
  }

  // Reserve memory for colors, shapes
  db.grids.clear();
  db.interiorColors.clear();
  db.exteriorColors.clear();
  db.shapes.clear();

  db.grids.reserve(db.gridRows * db.gridCols * shapeNum);
  db.interiorColors.reserve(shapeNum * 3);
  db.exteriorColors.reserve(shapeNum * 3);
  db.shapes.reserve(shapeNum);

  try {
    // Read shapes
    for (int s = 0; s < shapeNum; ++s) {
      // Get shape name
      Shape shape;
      std::getline(file, shape.name);

      // Get color
      for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        shape.color.interior[i] = std::stoi(line);
        db.interiorColors.push_back(shape.color.interior[i]);
      }
      for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        shape.color.exterior[i] = std::stoi(line);
        db.exteriorColors.push_back(shape.color.exterior[i]);
      }
      std::getline(file, line);
      shape.color.validInterior = std::stoi(line);
      std::getline(file, line);
      shape.color.validExterior = std::stoi(line);

      // Get areas
      std::getline(file, line);
      shape.area = std::stod(line);

      // Get bound
      std::getline(file, line);
      shape.bound.x = std::stoi(line);
      std::getline(file, line);
      shape.bound.y = std::stoi(line);
      std::getline(file, line);
      shape.bound.width = std::stoi(line);
      std::getline(file, line);
      shape.bound.height = std::stoi(line);

      // Get packed image
      PixelPack pack;
      db.packedImages.push_back(std::vector<PixelPack>());
      while (true) {
        std::getline(file, line);
        pack.type = std::stoi(line);
        std::getline(file, line);
        pack.repetitions = std::stoi(line);
        db.packedImages[s].push_back(pack);
        if (pack.repetitions == 0) {
          break;
        }
      }

      // Get grid
      for (int i = 0; i < db.gridRows * db.gridCols; ++i) {
        std::getline(file, line);
        int thisValue = std::stoi(line);
        db.grids.push_back(thisValue);
      }

      // Add shape
      db.shapes.emplace_back(shape);
    }
  } catch (...) {
    ROS_FATAL("Could not read a shape");
    ROS_BREAK();
    return;
  }

  // Read group number
  int groupNum;
  try {
    std::getline(file, line);
    groupNum = std::stoi(line);
  } catch (...) {
    ROS_FATAL("Could not read group count.");
    ROS_BREAK();
  }

  // Get groups
  db.groups.reserve(groupNum);
  try {
    for (int g = 0; g < groupNum; ++g) {
      ShapeGroup group;

      // Name
      std::getline(file, group.name);

      // Get pose
      for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        group.pose[i] = std::stod(line);
      }

      // Bound (only width & height)
      std::getline(file, line);
      group.bound.width = std::stoi(line);
      std::getline(file, line);
      group.bound.height = std::stoi(line);
      group.bound.x = 0;
      group.bound.y = 0;

      // Get indice num
      std::getline(file, line);
      int indiceNum = std::stoi(line);
      group.shapeIndices.reserve(indiceNum);

      // Get indices
      for (int i = 0; i < indiceNum; ++i) {
        std::getline(file, line);
        group.shapeIndices.push_back(std::stoi(line));
      }

      // Add group
      db.groups.emplace_back(group);
    }

  } catch (...) {
    ROS_FATAL("Could not read a group");
    ROS_BREAK();
    return;
  }

  // Success
  ROS_INFO("Successfully loaded %i shapes and %i groups!", db.shapes.size(),
           db.groups.size());
}

void saveShapeAnalysisDatabase(const std::string& filename, const ShapeDb& db) {
  // Open the file
  ROS_INFO("Writing shape analysis database at: %s", filename.c_str());
  std::ofstream file(filename, std::ios::out | std::ios::trunc);

  // Validate
  if (!file.good()) {
    ROS_FATAL("Could not open shape analysis database.");
    ROS_BREAK();
    return;
  }

  // Enable exceptions for easier error processing
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);

  try {
    // Header
    file << dbHeaderString << "\n";

    // Render params
    file << std::to_string(db.gridRows) << "\n";
    file << std::to_string(db.gridCols) << "\n";
    file << std::to_string(db.squareSize) << "\n";
    file << std::to_string(db.frameBufferHeight) << "\n";
    file << std::to_string(db.frameBufferWidth) << "\n";
    file << std::to_string(db.renderDistance) << "\n";

    // Shape count
    file << std::to_string(db.shapes.size()) << "\n";

    // Shapes
    // DB index is assumed to be in order
    for (int s = 0; s < db.shapes.size(); ++s) {
      // Name
      file << db.shapes[s].name << "\n";

      // Color
      for (int i = 0; i < 3; ++i) {
        file << std::to_string((int)db.shapes[s].color.interior[i]) << "\n";
      }
      for (int i = 0; i < 3; ++i) {
        file << std::to_string((int)db.shapes[s].color.exterior[i]) << "\n";
      }
      file << std::to_string((int)db.shapes[s].color.validInterior) << "\n";
      file << std::to_string((int)db.shapes[s].color.validExterior) << "\n";

      // Area
      file << std::to_string(db.shapes[s].area) << "\n";

      // Bound
      file << std::to_string((int)db.shapes[s].bound.x) << "\n";
      file << std::to_string((int)db.shapes[s].bound.y) << "\n";
      file << std::to_string((int)db.shapes[s].bound.width) << "\n";
      file << std::to_string((int)db.shapes[s].bound.height) << "\n";

      // Packed image
      for (int i = 0; i < db.packedImages[s].size(); ++i) {
        file << std::to_string(db.packedImages[s][i].type) << "\n";
        file << std::to_string(db.packedImages[s][i].repetitions) << "\n";
      }
      file << "0\n0\n";

      // Grid
      int initialGrid = db.gridRows * db.gridCols * s;
      for (int i = initialGrid; i < db.gridRows * db.gridCols + initialGrid;
           ++i) {
        file << std::to_string(db.grids[i]) << "\n";
      }
    }
  } catch (...) {
    ROS_FATAL("Something went wrong during the write.");
    ROS_BREAK();
    return;
  }

  // Group count
  file << std::to_string(db.groups.size()) << "\n";

  // Write groups
  for (int g = 0; g < db.groups.size(); ++g) {
    // Name
    file << db.groups[g].name << "\n";

    // Pose
    for (int i = 0; i < 3; ++i) {
      file << std::to_string(db.groups[g].pose[i]) << "\n";
    }

    // Bound (only width & height)
    file << std::to_string((int)db.groups[g].bound.width) << "\n";
    file << std::to_string((int)db.groups[g].bound.height) << "\n";

    // Shape indice count
    file << std::to_string(db.groups[g].shapeIndices.size()) << "\n";

    // Shape indices
    for (int i = 0; i < db.groups[g].shapeIndices.size(); ++i) {
      file << std::to_string(db.groups[g].shapeIndices[i]) << "\n";
    }
  }

  // Success
  ROS_INFO("Successfully wrote %i shapes and %i groups!", db.shapes.size(),
           db.groups.size());
}

}  // namespace au_vision