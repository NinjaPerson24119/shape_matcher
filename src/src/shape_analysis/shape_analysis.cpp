/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <au_vision/shape_analysis/shape_analysis_matcher.h>
#include <ros/ros.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <ctime>
#include <fstream>
#include <list>

namespace au_vision {

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
    if (lower[k] >= other.lower[k] && upper[k] <= other.upper[k]) {
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
                       ContourRenderer& renderer, const std::string& name,
                       const cv::Scalar& pose, const cv::Rect& groupBound) {
  if (!newShapes.size()) {
    return;
  }

  // Verify grids
  ROS_ASSERT(goodForAdd());

  // Verify renderer
  ROS_ASSERT(rendererIsCompatible(renderer));

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
    cv::Mat raster;
    renderer.render(s.contour, raster);

    // Kernel variables
    size_t gridArea = gridRows * gridCols;
    int* bins = new int[gridArea];
    int* dev_bins;
    gpuErrorCheck(cudaMalloc(&dev_bins, gridArea * sizeof(int)));
    unsigned char* dev_raster;
    size_t rasterBytes = sizeof(unsigned char) * raster.rows * raster.cols;
    gpuErrorCheck(cudaMalloc(&dev_raster, rasterBytes));
    gpuErrorCheck(cudaMemcpy(dev_raster, raster.data, rasterBytes,
                             cudaMemcpyHostToDevice));

    // Call kernel
    callBinRasterToGrid_device(dev_raster, gridRows, gridCols, squareSize,
                               dev_bins);

    // Retrieve bins
    gpuErrorCheck(cudaMemcpy(bins, dev_bins, gridArea * sizeof(int),
                             cudaMemcpyDeviceToHost));

    // Deallocate
    gpuErrorCheck(cudaFree(dev_bins));
    gpuErrorCheck(cudaFree(dev_raster));

    // Append the bins to the existing grids
    std::copy(bins, bins + gridArea, std::back_inserter(grids));

    // Add color
    for (int i = 0; i < 3; ++i) {
      colors.push_back(s.color[i]);
    }

    shapes.push_back(s);
  }
}

void ShapeDb::loadToGpu() {
  ROS_ASSERT(dev_grids == nullptr && dev_colors == nullptr);
  ROS_ASSERT(grids.size());
  ROS_ASSERT(colors.size());
  ROS_ASSERT(goodForDetect());

  // Upload grids to GPU
  size_t gridsBytes = gridRows * gridCols * shapes.size() * sizeof(int);
  gpuErrorCheck(cudaMalloc(&dev_grids, gridsBytes));
  gpuErrorCheck(
      cudaMemcpy(dev_grids, &grids[0], gridsBytes, cudaMemcpyHostToDevice));

  // Upload colors to GPU
  size_t colorsBytes = shapes.size() * sizeof(unsigned char) * 3;
  gpuErrorCheck(cudaMalloc(&dev_colors, colorsBytes));
  gpuErrorCheck(
      cudaMemcpy(dev_colors, &colors[0], colorsBytes, cudaMemcpyHostToDevice));
}

void ShapeDb::unloadFromGpu() {
  ROS_ASSERT(dev_grids != nullptr);

  gpuErrorCheck(cudaFree(dev_grids));
  gpuErrorCheck(cudaFree(dev_colors));

  dev_grids = nullptr;
  dev_colors = nullptr;
}

ShapeDb::~ShapeDb() {
  if (dev_grids != nullptr) {
    unloadFromGpu();
  }
}

bool ShapeDb::rendererIsCompatible(ContourRenderer& renderer) {
  if (renderer.width() != frameBufferWidth) {
    return false;
  }
  if (renderer.height() != frameBufferHeight) {
    return false;
  }
  return true;
}

bool ShapeDb::goodForAdd() {
  if (!gridCols) {
    return false;
  }
  if (!gridRows) {
    return false;
  }
  if (!frameBufferWidth) {
    return false;
  }
  if (!frameBufferHeight) {
    return false;
  }
  if (!squareSize) {
    return false;
  }
  if (frameBufferWidth % squareSize != 0) {
    return false;
  }
  if (frameBufferHeight % squareSize != 0) {
    return false;
  }
  return true;
}

bool ShapeDb::goodForDetect() {
  if (!goodForAdd()) {
    return false;
  }
  if (!shapes.size()) {
    return false;
  }
  if (!groups.size()) {
    return false;
  }
  if (grids.size() != shapes.size() * gridRows * gridCols) {
    return false;
  }
  return true;
}

void centerAndMaximizeContour(std::vector<cv::Point>& contour, int width,
                              int height) {
  // TODO: GPU accelerate

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
                  int imageHeight, double minArea) {
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
    int i = 0;
    while (true && contours.size() != 0) {
      if (contours[i].size() < 3) {
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

// Calculates the distance between two points
// TODO: Actually use this function for coherency testing
double calcDistance(const cv::Point& A, const cv::Point& B) {
  std::complex<double> a(A.x, A.y);
  std::complex<double> b(B.x, B.y);
  return sqrt(std::norm(b - a));
}

// Calculates the degrees angle between two points, with the x-axis as the
// reference
// TODO: Actually use this function for coherency testing
double calcVectorAngle(const cv::Point& A, const cv::Point& B) {
  double r = calcDistance(A, B);
  double x = std::abs(A.x - B.x);
  return std::acos(x / r) * 180 / PI;
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
    if (line != "SHAPE_ANALYSIS_DATABASE") {
      ROS_FATAL("Could not load shape analysis database. Invalid header.");
      ROS_BREAK();
    }
  } catch (...) {
    ROS_FATAL("Could not read header");
    ROS_BREAK();
  }

  // Read grid parameters
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
    ROS_FATAL("Could not read grid params");
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

  // Reserve memory for grids, shapes
  db.grids.reserve(db.gridRows * db.gridCols * shapeNum);
  db.colors.reserve(shapeNum * 3);
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
        shape.color[i] = std::stoi(line);
        db.colors.push_back(shape.color[i]);
      }

      // Get area
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

      // Get contour
      std::getline(file, line);
      int points = std::stoi(line);
      for (int i = 0; i < points; ++i) {
        std::getline(file, line);
        int thisX = std::stoi(line);

        std::getline(file, line);
        int thisY = std::stoi(line);

        cv::Point thisPoint(thisX, thisY);
        shape.contour.push_back(thisPoint);
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
    file << "SHAPE_ANALYSIS_DATABASE"
         << "\n";

    // Grid params
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
        file << std::to_string((int)db.shapes[s].color[i]) << "\n";
      }

      // Area
      file << std::to_string(db.shapes[s].area) << "\n";

      // Bound
      file << std::to_string((int)db.shapes[s].bound.x) << "\n";
      file << std::to_string((int)db.shapes[s].bound.y) << "\n";
      file << std::to_string((int)db.shapes[s].bound.width) << "\n";
      file << std::to_string((int)db.shapes[s].bound.height) << "\n";

      // Contour
      file << std::to_string(db.shapes[s].contour.size()) << "\n";
      for (int i = 0; i < db.shapes[s].contour.size(); ++i) {
        file << std::to_string(db.shapes[s].contour[i].x) << "\n";
        file << std::to_string(db.shapes[s].contour[i].y) << "\n";
      }

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