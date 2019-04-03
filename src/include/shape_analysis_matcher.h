/**
 * @author Nicholas Wengel
 */ 

#ifdef GPU

#ifndef AU_VISION_SHAPE_ANALYSIS_SHAPE_MATCHER
#define AU_VISION_SHAPE_ANALYSIS_SHAPE_MATCHER

#include <au_core/camera_info.h>
#include <au_vision/shape_analysis/shape_analysis.h>

#include <opencv2/opencv.hpp>

#include <map>
#include <vector>

namespace au_vision {

struct ShapeMatch {
  ShapeMatch() {}
  double rating;
  int rtIndex, dbIndex;
  cv::Rect bound;
};

struct ShapeGroupMatch {
  ShapeGroupMatch() {}
  double rating;
  double distance;
  int groupIndex;
  cv::Rect bound;
  std::vector<ShapeMatch> parts;
};

double groupAverageAreaRatio(
    const std::vector<std::vector<cv::Point>>& realtimeContours,
    const ShapeDb& db, const ShapeGroupMatch& match);

cv::Rect scaledGroupBound(const ShapeDb& db, const ShapeGroupMatch& groupMatch);

// Approximates the group distance
double approximateGroupDistance(
    std::vector<std::vector<cv::Point>>& realtimeContours, ShapeDb& db,
    ShapeGroupMatch& match, const au_core::CameraInfo& cameraInfo);

std::vector<ShapeGroupMatch> matchRealtimeContours(
    std::vector<std::vector<cv::Point>>& realtimeContours,
    std::vector<cv::Scalar>& realtimeColors, ShapeDb& db,
    MatchShapesThresholds& shapeTh, MatchShapeGroupThresholds& groupTh,
    ContourRenderer& renderer, std::vector<double>& bestLooseRatings,
    const au_core::CameraInfo& cameraInfo);

}  // namespace au_vision

#endif

#endif
