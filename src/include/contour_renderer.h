/**
 * @author Nicholas Wengel
 */ 

#ifdef GPU

#ifndef AU_VISION_CONTOUR_RENDERER
#define AU_VISION_CONTOUR_RENDERER

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace au_vision {

// Renders a contour to a binary image
class ContourRenderer {
 public:
  ContourRenderer();
  ~ContourRenderer();
  ContourRenderer(const ContourRenderer& other) = delete;
  ContourRenderer& operator=(const ContourRenderer& rhs) = delete;
  void initialize(int width, int height);
  void shutdown();
  int width() const {
    ROS_ASSERT(initialized_);
    return width_;
  }
  int height() const {
    ROS_ASSERT(initialized_);
    return height_;
  }
  void render(const std::vector<cv::Point> contour, cv::Mat& outMask);

 private:
  int width_, height_;
  bool initialized_;
};

}  // namespace au_vision

#endif

#endif