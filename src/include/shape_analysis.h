#ifdef GPU

#ifndef AU_VISION_SHAPE_ANALYSIS
#define AU_VISION_SHAPE_ANALYSIS

#include <au_vision/shape_analysis/superpixel_filter.h>

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace au_vision {

struct PixelPack {
  // the repeated bit pattern
  uint8_t type;
  // the number of times to repeat the bit pattern
  // a bit pattern cannot span more than a single row
  // this limits a bit pattern to 2^8-1, or
  // 255 (max length) * 8 (binary pixels per pack) = 2040 (max supported image
  // width)
  uint8_t repetitions;
};

// converts a vector of pixel packs into a compressed image
void packedToCompressed(const std::vector<PixelPack>& packedImage,
                        uint8_t* compressed);

// converts an array of bytes (each representing multiple binary pixels) into a
// list of repeating pixel packs
std::vector<PixelPack> compressedToPacked(const uint8_t* compressed,
                                          int sizeBytes, int imageWidth);

// Represents the color information of a shape
struct ShapeColor {
  cv::Scalar interior, exterior;
  bool validInterior, validExterior;
};

// Calculates the average interior and exterior colors of a contour
// filterSize is the "width" to dilate and erode by when calculating which
// pixels just outside and inside to take the average colors from
void averageContourColors(const cv::Mat& image,
                          const std::vector<std::vector<cv::Point>>& contours,
                          std::vector<ShapeColor>& colors, int filterSize,
                          cv::Mat* outSampleRegions = nullptr);

// Represents a range of colors
class ColorRange {
 public:
  ColorRange() : lower(cv::Scalar(0, 0, 0)), upper(cv::Scalar(0, 0, 0)) {}
  ColorRange(const cv::Scalar& target, int variance);
  bool contains(const ColorRange& other);
  cv::Scalar lower;
  cv::Scalar upper;
};

struct MatchShapesThresholds {
  int areaDifferenceThresh;
  int colorDifferenceThresh;
  int blackMargin, whiteMargin;
};

struct MatchShapeGroupThresholds {
  double minimumRating;
  double minimumShapeGroupRatio;
};

// Holds a shape
struct Shape {
  std::string name;
  ShapeColor color;
  std::vector<cv::Point>
      contour;     // these are not loaded nor saved by the DB functions
  double area;     // Of the unprocessed contour
  cv::Rect bound;  // Offset within group
};

struct ShapeGroup {
  std::string name;
  cv::Scalar pose;
  std::vector<int> shapeIndices;
  cv::Rect bound;
};

// Holds shape database
class ShapeDb {
 public:
  ShapeDb()
      : frameBufferWidth(0),
        frameBufferHeight(0),
        renderDistance(0.0),
        dev_grids(nullptr),
        dev_packedImages(nullptr),
        dev_interiorColors(nullptr),
        dev_exteriorColors(nullptr),
        loadedGpu(false) {}
  // only copies settings
  ShapeDb(const ShapeDb& other)
      : gridRows(other.gridRows),
        gridCols(other.gridCols),
        squareSize(other.squareSize),
        frameBufferWidth(other.frameBufferWidth),
        frameBufferHeight(other.frameBufferHeight),
        renderDistance(other.renderDistance),
        dev_grids(nullptr),
        dev_packedImages(nullptr),
        dev_interiorColors(nullptr),
        dev_exteriorColors(nullptr),
        loadedGpu(false) {}
  ~ShapeDb();
  void addGroup(const std::vector<Shape>& newShapes, const std::string& name,
                const cv::Scalar& pose, const cv::Rect& groupBound);
  void loadToGpu();
  void unloadFromGpu();
  std::vector<Shape> shapes;
  std::vector<unsigned char> interiorColors, exteriorColors;  // 3 components
  std::vector<std::vector<PixelPack>> packedImages;
  std::vector<int> grids;
  std::vector<ShapeGroup> groups;
  int gridRows, gridCols, squareSize, frameBufferWidth, frameBufferHeight;
  int* dev_grids;
  PixelPack* dev_packedImages;
  int maxPacks;
  unsigned char* dev_interiorColors;
  unsigned char* dev_exteriorColors;
  double renderDistance;
  bool loadedGpu;
};

// Scales a contour so that it is as large as possible within a certain frame
// size, while maintaining aspect ratio  The shape is also centered on the frame
// dimensions
void centerAndMaximizeContour(std::vector<cv::Point>& contour, int width,
                              int height);

// Culls contours from the passed vector using the following criteria:
// Contours must have a minimum area
// Contours must not be touching the image edge
// Contours must have at least 3 points
// NOTE: gSLICr will rescale images, so use its image dimensions rather than the
// preprocessed dimensions
void cullContours(std::vector<std::vector<cv::Point>>& contours, int imageWidth,
                  int imageHeight, double minArea, int minPoints,
                  int maxPoints);

// Loads a database to be used for shape matching
void loadShapeAnalysisDatabase(const std::string& filename, ShapeDb& db);

// Writes a database to be used for shape matching
void saveShapeAnalysisDatabase(const std::string& filename, const ShapeDb& db);

}  // namespace au_vision

#endif

#endif
