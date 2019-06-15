#ifdef GPU

#ifndef AU_VISION_SHAPE_ANALYSIS_COMMON
#define AU_VISION_SHAPE_ANALYSIS_COMMON

#include <au_core/vision_util.h>
#include <au_vision/detector.h>

#include <au_vision/shape_analysis/auto_filter.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <au_vision/shape_analysis/superpixel_filter.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace au_vision {

class ShapeDetectorCommon : public Detector {
 public:
  ShapeDetectorCommon(ros::NodeHandle& nh, ros::NodeHandle& private_nh,
                      std::string detectorType);
  ~ShapeDetectorCommon();
  ShapeDetectorCommon(const ShapeDetectorCommon& other) = delete;
  ShapeDetectorCommon& operator=(const ShapeDetectorCommon& rhs) = delete;

 protected:
  std::vector<au_core::Roi> detect(const cv::Mat& input,
                                   const au_core::CameraInfo& cameraInfo);
  SuperPixelFilter spfilter_;
  ShapeDb db_;
  MatchShapesThresholds shapeThresholds_;
  MatchShapeGroupThresholds groupThresholds_;
  bool initialized_, visualizeAutoFilterHistogram_;
  int blackMargin_, whiteMargin_, autoFilterBins_, minContourArea_, minPoints_,
      maxPoints_, morphologyFilterSize_;
  double contourLinearizationEpsilon_;
  int parallelFrames_;
  std::string name_;
};

}  // namespace au_vision

#endif

#endif