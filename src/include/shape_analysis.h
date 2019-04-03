/**
 * @author Nicholas Wengel
 */ 

#ifdef GPU

#ifndef AU_VISION_SHAPE_ANALYSIS
#define AU_VISION_SHAPE_ANALYSIS

#include <au_vision/shape_analysis/contour_renderer.h>
#include <au_vision/shape_analysis/superpixel_filter.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace au_vision {

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
};

struct MatchShapeGroupThresholds {
  double minimumRating;
  // double instantAngleThresh;
  // double instantDistanceThresh;
  // double cumulativeAngleThresh;
  // double cimulativeDistanceThresh;
};

struct CollisionRemovalThresholds {
  int minAngleDiffThresh;  // Minimum angle difference to consider removal
  double minRatingThresh;  // Minimum rating to consider removal
};

// Holds a shape
struct Shape {
  std::string name;
  cv::Scalar color;
  double area;                     // Of the unprocessed contour
  std::vector<cv::Point> contour;  // Shape contour points
  cv::Rect bound;                  // Offset within group
};

struct ShapeGroup {
  std::string name;
  cv::Scalar pose;
  std::vector<int> shapeIndices;
  cv::Rect bound;
  // std::vector<std::vector<double>> angles;
  // std::vector<std::vector<double>> distances;
};

// Holds shape database
class ShapeDb {
 public:
  ShapeDb()
      : gridRows(0),
        gridCols(0),
        squareSize(0),
        frameBufferWidth(0),
        frameBufferHeight(0),
        renderDistance(0.0),
        dev_grids(nullptr),
        dev_colors(nullptr) {}
  ShapeDb(ShapeDb& other)
      : gridRows(other.gridRows),
        gridCols(other.gridCols),
        squareSize(other.squareSize),
        frameBufferWidth(other.frameBufferWidth),
        frameBufferHeight(other.frameBufferHeight),
        renderDistance(other.renderDistance),
        dev_grids(nullptr),
        dev_colors(nullptr) {}
  ~ShapeDb();
  void addGroup(const std::vector<Shape>& newShapes, ContourRenderer& renderer,
                const std::string& name, const cv::Scalar& pose,
                const cv::Rect& groupBound);
  void loadToGpu();
  void unloadFromGpu();
  bool rendererIsCompatible(ContourRenderer& renderer);
  bool goodForAdd();
  bool goodForDetect();
  std::vector<Shape> shapes;
  std::vector<unsigned char> colors;  // 3 component
  std::vector<int> grids;
  std::vector<ShapeGroup> groups;
  int gridRows, gridCols, squareSize, frameBufferWidth, frameBufferHeight;
  int* dev_grids;
  unsigned char* dev_colors;
  double renderDistance;
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
void cullContours(std::vector<std::vector<cv::Point> >& contours,
                  int imageWidth, int imageHeight, double minArea);

// Calculates the distance between two points
double calcDistance(const cv::Point& A, const cv::Point& B);

// Calculates the degrees angle between two points, with the x-axis as the
// reference
double calcVectorAngle(const cv::Point& A, const cv::Point& B);

// Loads a database to be used for shape matching
void loadShapeAnalysisDatabase(const std::string& filename, ShapeDb& db);

// Writes a database to be used for shape matching
void saveShapeAnalysisDatabase(const std::string& filename, const ShapeDb& db);

// Uses the passed filters and an image to add shapes and grids to the passed
// variables
// Returns the number of new shapes
int calcShapesFromImage(const cv::Mat& imageRgb,
                        const std::vector<ColorRange>& filters,
                        SuperPixelFilter& spFilter, ContourRenderer& renderer,
                        int rotX, int rotY, int rotZ, const std::string& name,
                        cv::Mat& viewOut, cv::Mat& viewOutProcessed,
                        bool gSLICrOn, ShapeDb& db);

// Removes similar shapes with different angles
// void removeDatabaseCollisions(std::vector<Shape>& shapes, ShapeGrids& grids,
// ContourRenderer& renderer, MatchShapesThresholds& matchTh,
// CollisionRemovalThresholds& collisionTh);

}  // namespace au_vision

#endif

#endif
