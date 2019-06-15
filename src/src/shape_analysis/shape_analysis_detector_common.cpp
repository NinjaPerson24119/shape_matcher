#include <au_vision/shape_analysis/shape_analysis_detector_common.h>

#include <au_vision/shape_analysis/auto_filter.h>
#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <au_vision/shape_analysis/shape_analysis_matcher.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

#include <au_core/math_util.h>
#include <ros/package.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <set>
#include <sstream>
#include <thread>

// DEBUG TODO
#include <ctime>

namespace au_vision {

ShapeDetectorCommon::ShapeDetectorCommon(ros::NodeHandle& nh,
                                         ros::NodeHandle& private_nh,
                                         std::string detectorType)
    : Detector(nh, private_nh, detectorType) {
  // Get db absolute path
  std::string packagePath = ros::package::getPath("au_vision");
  std::string dbName;
  auto prefix = detectorType + "/";
  if (!private_nh_.getParam(prefix + "database", dbName)) {
    ROS_FATAL("Shape Detector: missing database name");
    ROS_BREAK();
  }
  std::string dbPath = packagePath + "/shape_dbs/" + dbName;

  // Debug topics
  addCustomDebugImageTopic("debug_GraySuperpixels");
  addCustomDebugImageTopic("debug_LineOverlay");
  addCustomDebugImageTopic("debug_AverageColorSuperpixels");
  addCustomDebugImageTopic("debug_BinaryMaskWithHud");
  addCustomDebugImageTopic("debug_InputStageContours");
  addCustomDebugImageTopic("debug_AutoFilterMask");
  addCustomDebugImageTopic("debug_AutoFilterHistogram");
  addCustomDebugImageTopic("debug_Edges");
  addCustomDebugImageTopic("debug_AverageColorSampleRegions");

  // Initialize spfilter
  spfilter_.initialize(private_nh, prefix);

  // Get detection settings
  if (!private_nh.getParam(prefix + "morphology_filter_size",
                           morphologyFilterSize_) ||
      !private_nh.getParam(prefix + "min_points", minPoints_) ||
      !private_nh.getParam(prefix + "max_points", maxPoints_) ||
      !private_nh.getParam(prefix + "minimum_shape_group_ratio",
                           groupThresholds_.minimumShapeGroupRatio) ||
      !private_nh.getParam(prefix + "area_difference_thresh",
                           shapeThresholds_.areaDifferenceThresh) ||
      !private_nh.getParam(prefix + "color_difference_thresh",
                           shapeThresholds_.colorDifferenceThresh) ||
      !private_nh.getParam(prefix + "minimum_rating",
                           groupThresholds_.minimumRating) ||
      !private_nh.getParam(prefix + "auto_filter_bins", autoFilterBins_) ||
      !private_nh.getParam(prefix + "auto_filter_white_margin", whiteMargin_) ||
      !private_nh.getParam(prefix + "auto_filter_black_margin", blackMargin_) ||
      !private_nh.getParam(prefix + "auto_filter_histogram",
                           visualizeAutoFilterHistogram_) ||
      !private_nh.getParam(prefix + "minimum_contour_area", minContourArea_) ||
      !private_nh.getParam(prefix + "contour_linearization_epsilon",
                           contourLinearizationEpsilon_)) {
    ROS_FATAL("ShapeDetector: missing a detection parameter");
    ROS_BREAK();
  }

  shapeThresholds_.blackMargin = blackMargin_;
  shapeThresholds_.whiteMargin = whiteMargin_;

  if (morphologyFilterSize_ % 2 == 0) {
    ROS_FATAL("Morphology filter size must be odd");
    ROS_BREAK();
  }

  // Check CUDA and OpenCV
  initCudaAndOpenCv();

  // Load shape analysis database
  loadShapeAnalysisDatabase(dbPath, db_);

  // estimate memory usage
  // we're only really considering the number of frames to allocate for shape
  // comparison parallelization there's probably at least ~300 MB extra on both
  // VRAM and RAM from other things
  const size_t maxVRAM = 0.5e9;

  size_t compImageBytes = db_.frameBufferWidth * db_.frameBufferHeight / 8;
  size_t bytesPerParallelFrame =
      compImageBytes * 2;  // VRAM (2x for unpack and result)
  parallelFrames_ = maxVRAM / bytesPerParallelFrame;

  ROS_ASSERT(parallelFrames_);
  ROS_INFO("VRAM is roughly maxed at: %0.2f MB", (double)maxVRAM / 1.0e6);
  ROS_INFO("Parallel Frames (Compressed [dims / 8] 2x): %i", parallelFrames_);

  // Load DB to gpu
  db_.loadToGpu();
}

ShapeDetectorCommon::~ShapeDetectorCommon() {}

std::vector<au_core::Roi> ShapeDetectorCommon::detect(
    const cv::Mat& input, const au_core::CameraInfo& cameraInfo) {
  // Make a copy so that the input retains its color gamut for reference
  cv::Mat lab = input.clone();

  // Convert input to Lab colorspace
  cv::cuda::GpuMat dev_lab(lab);
  cv::cuda::cvtColor(dev_lab, dev_lab, cv::COLOR_RGB2Lab);
  dev_lab.download(lab);

  // Unsharp mask
  cv::cuda::GpuMat dev_blur;
  cv::Ptr<cv::cuda::Filter> gFilter =
      cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(0, 0), 3);
  gFilter->apply(dev_lab, dev_blur);
  cv::cuda::addWeighted(dev_lab, 1.5, dev_blur, -0.5, 0, dev_lab);
  dev_lab.download(lab);

  // Filter image
  cv::Mat mask, overlay, colorMask;
  spfilter_.filterLabImage(lab);
  spfilter_.resultMask(mask, true);

  if (debug_) {
    spfilter_.resultLineOverlay(overlay);
  }

  // Get solid color spixels
  std::vector<cv::Scalar> colorList;
  spfilter_.resultColors(colorList);
  spfilter_.resultAverageColorMask(colorMask, colorList);

  // Use auto filter for non-shades
  cv::Mat autoFilterHist, autoFilterMask, unscaledAutoFilterMask;
  std::vector<cv::Mat> contourMasks;

  maskFromAutoFilter(
      colorMask, colorList, autoFilterBins_, blackMargin_, whiteMargin_,
      visualizeAutoFilterHistogram_ && debug_ ? &autoFilterHist : nullptr,
      autoFilterMask);

  // Edge detect auto filter mask
  cv::Mat edges(autoFilterMask.rows, autoFilterMask.cols, CV_8UC1);

  unsigned short* dev_autoFilterMask;
  unsigned char* dev_edges;
  gpuErrorCheck(cudaMalloc(
      &dev_autoFilterMask,
      sizeof(unsigned short) * autoFilterMask.rows * autoFilterMask.cols));
  gpuErrorCheck(cudaMalloc(
      &dev_edges,
      sizeof(unsigned char) * autoFilterMask.rows * autoFilterMask.cols));
  gpuErrorCheck(cudaMemcpy(
      dev_autoFilterMask, autoFilterMask.data,
      sizeof(unsigned short) * autoFilterMask.rows * autoFilterMask.cols,
      cudaMemcpyHostToDevice));

  callSimpleEdgeDetect_device(dev_autoFilterMask, dev_edges,
                              autoFilterMask.rows, autoFilterMask.cols);

  gpuErrorCheck(cudaMemcpy(
      edges.data, dev_edges,
      sizeof(unsigned char) * autoFilterMask.rows * autoFilterMask.cols,
      cudaMemcpyDeviceToHost));
  gpuErrorCheck(cudaFree(dev_autoFilterMask));
  gpuErrorCheck(cudaFree(dev_edges));

  // Find contours (no GPU function available, so using CPU)
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

  // Draw input stage contours
  cv::Mat inputContoursDisplay(colorMask.rows, colorMask.cols, CV_8UC3,
                               cv::Scalar(0, 0, 0));
  if (debug_) {
    cv::drawContours(inputContoursDisplay, contours, -1, cv::Scalar(255, 0, 0),
                     4);
  }

  // Linearize contours
  for (int i = 0; i < contours.size(); ++i) {
    // Higher epsilon is more linearized
    cv::approxPolyDP(contours[i], contours[i], contourLinearizationEpsilon_,
                     true);
  }

  // Cull contours
  // This is necessary to ensure that each contour has at least three points
  // Otherwise the shape matching function will assert
  cullContours(contours, spfilter_.width(), spfilter_.height(), minContourArea_,
               minPoints_, maxPoints_);

  // Find interior and exterior contour colors
  std::vector<ShapeColor> colors;
  cv::Mat averageColorSampleRegions(colorMask.rows, colorMask.cols, CV_8UC1,
                                    cv::Scalar(0));
  averageContourColors(colorMask, contours, colors, morphologyFilterSize_,
                       debug_ ? &averageColorSampleRegions : nullptr);

  // Get matches
  std::vector<double> looseMatches;
  std::vector<ShapeGroupMatch> matches = matchRealtimeContours(
      contours, colors, db_, shapeThresholds_, groupThresholds_, looseMatches,
      cameraInfo, parallelFrames_);

  if (debug_) {
    // convert colormask to RGB
    cv::cuda::GpuMat dev_colorMask(colorMask);
    cv::cuda::cvtColor(dev_colorMask, dev_colorMask, cv::COLOR_Lab2RGB);
    dev_colorMask.download(colorMask);

    // Draw contours and points (just overwrite, since it probably takes less
    // time than sorting)
    cv::Mat binaryHud(colorMask.rows, colorMask.cols, CV_8UC3,
                      cv::Scalar(0, 0, 0));
    cv::drawContours(binaryHud, contours, -1, cv::Scalar(255, 0, 0), 4);

    // Loose contours should overwrite overall outlines
    std::vector<std::vector<cv::Point>> looseContours;
    for (int i = 0; i < looseMatches.size(); ++i) {
      if (looseMatches[i]) {
        looseContours.push_back(contours[i]);
      }
    }
    cv::drawContours(binaryHud, looseContours, -1, cv::Scalar(0, 0, 255), 4);

    // Good contours should overwrite overall outlines & loose outlines
    for (auto m : matches) {
      std::vector<std::vector<cv::Point>> temp;
      for (int i = 0; i < m.parts.size(); ++i) {
        temp.push_back(contours[m.parts[i].rtIndex]);
      }
      cv::drawContours(binaryHud, temp, -1, cv::Scalar(0, 255, 0), 4);
    }

    for (int i = 0; i < contours.size(); ++i) {
      for (int j = 0; j < contours[i].size(); ++j) {
        cv::circle(binaryHud, contours[i][j], 3,
                   colorMask.at<cv::Vec3b>(contours[i][j].y, contours[i][j].x),
                   3);
      }
    }

    // Draw match ratings
    std::set<int> matchRtIndices;
    for (auto m : matches) {
      for (int i = 0; i < m.parts.size(); ++i) {
        std::stringstream ss;
        ss << "R: " << std::setprecision(2) << m.parts[i].rating;
        cv::putText(binaryHud, ss.str(), contours[m.parts[i].rtIndex][0],
                    cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(255, 200, 0), 2);
        matchRtIndices.insert(m.parts[i].rtIndex);
      }
    }
    for (int i = 0; i < looseMatches.size(); ++i) {
      if (looseMatches[i]) {
        auto exists = matchRtIndices.find(i);
        if (exists == matchRtIndices.end()) {
          std::stringstream ss;
          ss << "R: " << std::setprecision(2) << looseMatches[i];
          cv::putText(binaryHud, ss.str(), contours[i][0],
                      cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(255, 200, 0),
                      2);
        }
      }
    }

    // Debug outputs
    cv::cuda::GpuMat dev_overlay(overlay);
    cv::cuda::cvtColor(dev_overlay, dev_overlay, cv::COLOR_Lab2RGB);
    dev_overlay.download(overlay);

    // Send images to feeds...

    // These are the grey numbered superpixels
    publishCustomDebugImage("mono16", "debug_GraySuperpixels", mask);

    // Wireframe superpixels
    publishCustomDebugImage("rgb8", "debug_LineOverlay", overlay);

    // Averaged superpixels
    publishCustomDebugImage("rgb8", "debug_AverageColorSuperpixels", colorMask);

    // This is the black and white threshold output
    publishCustomDebugImage("rgb8", "debug_BinaryMaskWithHud", binaryHud);

    // This outputs the input contours
    publishCustomDebugImage("rgb8", "debug_InputStageContours",
                            inputContoursDisplay);

    // Auto filter gray chunks
    publishCustomDebugImage("mono16", "debug_AutoFilterMask", autoFilterMask);

    // Auto filter histograms
    if (visualizeAutoFilterHistogram_) {
      publishCustomDebugImage("rgb8", "debug_AutoFilterHistogram",
                              autoFilterHist);
    }

    // Edges used to build contours
    publishCustomDebugImage("mono8", "debug_Edges", edges);

    // Regions used to calculate average colors
    publishCustomDebugImage("mono8", "debug_AverageColorSampleRegions",
                            averageColorSampleRegions);
  }

  // Create ROIs for successful matches
  std::vector<au_core::Roi> rois;
  for (auto m : matches) {
    // Output parts info
    for (int i = 0; i < m.parts.size(); ++i) {
      au_core::Roi tempRoi;

      // No name to avoid clutter
      tempRoi.tags.push_back("");

      // Bound
      cv::Rect bound = cv::boundingRect(contours[m.parts[i].rtIndex]);
      tempRoi.topLeft.x = bound.x;
      tempRoi.topLeft.y = bound.y;
      tempRoi.width = bound.width;
      tempRoi.height = bound.height;

      // Pose
      Eigen::Vector3d rpy;
      for (int i = 0; i < 3; ++i) {
        rpy[i] = au_core::degToRad(db_.groups[m.groupIndex].pose[i]);
      }
      Eigen::Quaterniond q = au_core::rpyToQuat(rpy);

      // Pose
      tempRoi.poseEstimate.orientation.x = q.x();
      tempRoi.poseEstimate.orientation.y = q.y();
      tempRoi.poseEstimate.orientation.z = q.z();
      tempRoi.poseEstimate.orientation.w = q.w();
      tempRoi.poseEstimate.position.z = m.distance;
      tempRoi.hasOrientation = 1;

      // Confidence
      tempRoi.confidence = m.rating;
      tempRoi.heightConfidence = 1;
      tempRoi.widthConfidence = 1;

      // Add polygon
      for (auto pt : contours[m.parts[i].rtIndex]) {
        geometry_msgs::Point tempPoint;
        tempPoint.x = pt.x;
        tempPoint.y = pt.y;
        tempRoi.polygon.push_back(tempPoint);
      }

      // Adjust ROI based on spfilter
      spfilter_.adjustRoi(tempRoi, input);

      rois.push_back(tempRoi);
    }

    // Add group ROI
    au_core::Roi tempRoi;

    // Name / Drawn info
    std::stringstream ss;
    ss << db_.groups[m.groupIndex].name << " R: " << std::setprecision(2)
       << m.rating
       << " X: " << std::to_string((int)db_.groups[m.groupIndex].pose[0])
       << " Y: " << std::to_string((int)db_.groups[m.groupIndex].pose[1])
       << " Z: " << std::to_string((int)db_.groups[m.groupIndex].pose[2])
       << std::setprecision(2) << " D: " << m.distance;
    tempRoi.tags.push_back(ss.str());

    // Tag with just the name
    tempRoi.tags.push_back(db_.groups[m.groupIndex].name);

    // Confidence
    tempRoi.confidence = m.rating;
    tempRoi.heightConfidence = 1;
    tempRoi.widthConfidence = 1;

    // Bound
    tempRoi.topLeft.x = m.bound.x;
    tempRoi.topLeft.y = m.bound.y;
    tempRoi.width = m.bound.width;
    tempRoi.height = m.bound.height;

    // Pose
    Eigen::Vector3d rpy;
    for (int i = 0; i < 3; ++i) {
      rpy[i] = au_core::degToRad(db_.groups[m.groupIndex].pose[i]);
    }
    Eigen::Quaterniond q = au_core::rpyToQuat(rpy);

    // Pose
    tempRoi.poseEstimate.orientation.x = q.x();
    tempRoi.poseEstimate.orientation.y = q.y();
    tempRoi.poseEstimate.orientation.z = q.z();
    tempRoi.poseEstimate.orientation.w = q.w();
    tempRoi.poseEstimate.position.z = m.distance;
    tempRoi.hasOrientation = 1;

    // Adjust ROI based on spfilter
    spfilter_.adjustRoi(tempRoi, input);

    rois.push_back(tempRoi);
  }

  // Return
  return rois;
}

}  // namespace au_vision