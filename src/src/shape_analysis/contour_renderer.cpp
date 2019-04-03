/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/contour_renderer.h>

namespace au_vision {

ContourRenderer::ContourRenderer() : initialized_(false) {}

ContourRenderer::~ContourRenderer() {
  if (initialized_) {
    shutdown();
  }
}

void ContourRenderer::initialize(int width, int height) {
  ROS_ASSERT(initialized_ == false);
  width_ = width;
  height_ = height;
  initialized_ = true;
}

void ContourRenderer::shutdown() {
  ROS_ASSERT(initialized_ == true);
  initialized_ = false;
}

void ContourRenderer::render(const std::vector<cv::Point> contour,
                             cv::Mat& outMask) {
  // TODO: GPU accelerate (try this https://github.com/mapbox/earcut.hpp with
  // OpenGL FBO)

  outMask = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));
  std::vector<std::vector<cv::Point>> contours;
  contours.push_back(contour);
  cv::drawContours(outMask, contours, -1, cv::Scalar(255), CV_FILLED);
}

}  // namespace au_vision