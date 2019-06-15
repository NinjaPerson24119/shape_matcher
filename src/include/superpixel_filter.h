#ifdef GPU

#ifndef AU_VISION_SUPERPIXEL_FILTER_H
#define AU_VISION_SUPERPIXEL_FILTER_H

#include <au_vision/shape_analysis/superpixel_filter_kernels.h>

#include <gSLICr_Lib/gSLICr.h>

#include <ros/ros.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <au_core/Roi.h>

namespace au_vision {

// RAII for gSLICr super pixel segmentation
// NOTE: This class is thread safe (with the exception of getter functions,
// which might tear if the init func is abused)
class SuperPixelFilter {
 public:
  SuperPixelFilter();
  ~SuperPixelFilter();
  SuperPixelFilter(const SuperPixelFilter& other) = delete;
  SuperPixelFilter& operator=(const SuperPixelFilter& rhs) = delete;

  // Initializes the filter for a single image size
  // Uses a config file to override default values
  // The dimension override variables provide a mechanism to programmatically
  // change the filter size. These values will override values in the config
  // file.
  void initialize(ros::NodeHandle& handle, std::string nsPrefix = "",
                  int overrideWidth = -1, int overrideHeight = -1);

  // Used to check if the filter has been initialized
  bool isInitialized() const { return engine_ != nullptr; }

  // Returns the image width that the filter was initialized for
  int width() const { return width_; }

  // Returns the image height that the filter was initialized for
  int height() const { return height_; }

  // Filters a Lab format image. Results are obtained using other functions.
  // Note that images which do not match the initialized size will be resized
  void filterLabImage(const cv::Mat& input);

  // Returns a greyscale image where each superpixel is colored a different
  // shade  The returned mask will be the format CV_16SC1, which can be passed
  // to CvBridge::CvImage using the encoding "mono16".  If the result will be
  // displayed, set normalized to true to spread values out (making it easier to
  // notice differences). Returns the number of superpixels as an int
  int resultMask(cv::Mat& out, bool scaleForContrast = true);

  // Returns the original image with superpixel edges overlayed upon it
  // The returned image will be in Lab format, which can be passed to
  // CvBridge::CvImage using the encoding "rgb8" (after conversion with OpenCV)
  void resultLineOverlay(cv::Mat& out);

  // Returns a color image where each superpixel is colored with its average
  // color from the original image
  void resultAverageColorMask(cv::Mat& colorMaskOut,
                              const std::vector<cv::Scalar>& inColorList);

  // Returns a vector of the superpixel colors (the fourth channel indicates the
  // source spixel's gray value)
  void resultColors(std::vector<cv::Scalar>& colorList);

  // Takes in the image being processed and the region of interest message.
  // Adjusts roi if needed.
  void adjustRoi(au_core::Roi& roi, const cv::Mat& image) const;

 private:
  // These convert between gSLICr image types and OpenCV types
  void convertImage(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
  void convertImage(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);

  // This specialized version returns the number of superpixels
  int convertImage(const gSLICr::IntImage* inimg, cv::Mat& outimg,
                   bool scaleForContrast);

  // Data
  int resizeWarnWidth_, resizeWarnHeight_;
  int width_, height_;
  std::unique_ptr<gSLICr::engines::core_engine> engine_;
  std::unique_ptr<gSLICr::UChar4Image> colorImage_,
      overlayImage_;  // colorImage holds the original image in vector form;
                      // overlayImage is a buffer for drawing without modifying
                      // colorImage
  cudaTextureObject_t grayTex_,
      colorTex_;  // texture objects used for texture caching in kernels
  int* dev_grayImageData_;  // holds the gray mask image on the device (note
                            // that prefix "dev" means the pointer is allocated
                            // on the device)
  std::unique_ptr<gSLICr::Vector4i[]>
      spixelBins_;                    // holds the average color for each spixel
  gSLICr::Vector4i* dev_spixelBins_;  // device counterpart
  gSLICr::Vector4u* dev_colorImageData_;  // device counterpart (also used for
                                          // conversions though, OK to
                                          // overwrite)
  unsigned char* dev_openCvImageData_;  // holds the OpenCV matrix format of an
                                        // image during conversions to and from
                                        // vector format
  size_t colorImageBufferBytes_,
      openCvImageBufferBytes_;    // sizes of the device and host buffers
  int blocks_, threadsPerBlock_;  // device info for calling kernels
  bool initialFilterDone_,
      initialColorAverageDone_;  // flags for setup and tear down
};

}  // namespace au_vision

#endif

#endif
