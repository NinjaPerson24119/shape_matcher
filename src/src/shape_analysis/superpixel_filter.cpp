#include <au_vision/shape_analysis/superpixel_filter.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace au_vision {

// Constructor
SuperPixelFilter::SuperPixelFilter()
    : engine_(nullptr),
      colorImage_(nullptr),
      overlayImage_(nullptr),
      dev_grayImageData_(nullptr),
      spixelBins_(nullptr),
      dev_spixelBins_(nullptr),
      dev_colorImageData_(nullptr),
      dev_openCvImageData_(nullptr),
      initialFilterDone_(false),
      initialColorAverageDone_(false),
      resizeWarnWidth_(-1),
      resizeWarnHeight_(-1) {
  // Check CUDA and OpenCV
  initCudaAndOpenCv();
  blocks_ = global_totalSmBlocks;
  threadsPerBlock_ = global_threadsPerBlock;
}

SuperPixelFilter::~SuperPixelFilter() {
  // Free allocated memory
  if (initialColorAverageDone_) {
    // Destroy textures
    gpuErrorCheck(cudaDestroyTextureObject(colorTex_));
    gpuErrorCheck(cudaDestroyTextureObject(grayTex_));

    gpuErrorCheck(cudaFree(dev_spixelBins_));
    gpuErrorCheck(cudaFree(dev_grayImageData_));
  }
  if (dev_colorImageData_ != nullptr) {
    gpuErrorCheck(cudaFree(dev_colorImageData_));
  }
  if (dev_openCvImageData_ != nullptr) {
    gpuErrorCheck(cudaFree(dev_openCvImageData_));
  }
}

// Initialization
void SuperPixelFilter::initialize(ros::NodeHandle& handle, std::string nsPrefix,
                                  int overrideWidth, int overrideHeight) {
  if (engine_) {
    engine_.reset(nullptr);
  }

  if (initialColorAverageDone_) {
    gpuErrorCheck(cudaDestroyTextureObject(colorTex_));
    gpuErrorCheck(cudaDestroyTextureObject(grayTex_));

    gpuErrorCheck(cudaFree(dev_spixelBins_));
    gpuErrorCheck(cudaFree(dev_colorImageData_));
    gpuErrorCheck(cudaFree(dev_grayImageData_));
    gpuErrorCheck(cudaFree(dev_openCvImageData_));
    initialColorAverageDone_ = false;
  }

  // Read config values
  gSLICr::objects::settings config;

  // This setting is mandatory
  // NOTE: do not set this to gSLICr::CIELAB as that will make gSLICr convert
  // our already CIELAB image again into CIELAB By setting it to RGB, gSLICr
  // just skips the conversion, which is ideal.
  config.color_space = gSLICr::RGB;

  // Others...
  if (!handle.getParam(nsPrefix + "gSLICr_width", config.img_size.x)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSCLIr_width'");
    ROS_BREAK();
  }
  if (!handle.getParam(nsPrefix + "gSLICr_height", config.img_size.y)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSCLIr_height'");
    ROS_BREAK();
  }
  if (!handle.getParam(nsPrefix + "gSLICr_noSegs", config.no_segs)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSLICr_noSegs'");
    ROS_BREAK();
  }
  if (!handle.getParam(nsPrefix + "gSLICr_spixelSize", config.spixel_size)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSLICr_spixelSize'");
    ROS_BREAK();
  }
  int sizeInsteadOfNumber;
  if (!handle.getParam(nsPrefix + "gSLICr_sizeInsteadOfNumber",
                       sizeInsteadOfNumber)) {
    ROS_FATAL(
        "SuperPixelFilter: could not get parameter "
        "'gSLICr_sizeInsteadOfNumber'");
    ROS_BREAK();
  }
  config.seg_method = static_cast<gSLICr::SEG_METHOD>(sizeInsteadOfNumber);
  if (!handle.getParam(nsPrefix + "gSLICr_cohWeight", config.coh_weight)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSLICr_cohWeight'");
    ROS_BREAK();
  }
  if (!handle.getParam(nsPrefix + "gSLICr_noIters", config.no_iters)) {
    ROS_FATAL("SuperPixelFilter: could not get parameter 'gSLICr_noIters'");
    ROS_BREAK();
  }
  if (!handle.getParam(nsPrefix + "gSLICr_doEnforceConnectivity",
                       config.do_enforce_connectivity)) {
    ROS_FATAL(
        "SuperPixelFilter: could not get parameter "
        "'gSLICr_doEnforceConnectivity'");
    ROS_BREAK();
  }

  // Override dimensions if need be
  if (overrideWidth != -1) {
    ROS_ASSERT(overrideWidth > 0);
    config.img_size.x = overrideWidth;
  }
  if (overrideHeight != -1) {
    ROS_ASSERT(overrideHeight > 0);
    config.img_size.y = overrideHeight;
  }

  width_ = config.img_size.x;
  height_ = config.img_size.y;

  // Allocate engine
  engine_.reset(new gSLICr::engines::core_engine(config));

  // Allocate on both CPU and GPU
  gSLICr::Vector2i dimensions;
  dimensions.x = config.img_size.x;
  dimensions.y = config.img_size.y;
  colorImage_.reset(new gSLICr::UChar4Image(dimensions, true, true));
  overlayImage_.reset(new gSLICr::UChar4Image(dimensions, true, true));

  // Set up image conversion materials for CUDA
  colorImageBufferBytes_ =
      sizeof(gSLICr::Vector4u) * (config.img_size.x * config.img_size.y);
  openCvImageBufferBytes_ =
      config.img_size.x * config.img_size.y * 3;  // We assume 3 channels
  gpuErrorCheck(cudaMalloc(&dev_colorImageData_, colorImageBufferBytes_));
  gpuErrorCheck(cudaMalloc(&dev_openCvImageData_, openCvImageBufferBytes_));

  // Register change
  initialFilterDone_ = false;
}

// Converts from OpenCV matrix to gSLICr image
void SuperPixelFilter::convertImage(const cv::Mat& inimg,
                                    gSLICr::UChar4Image* outimg) {
  ROS_ASSERT(outimg);
  ROS_ASSERT(inimg.isContinuous());

  // Get pointers
  gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
  unsigned char* inimg_ptr = inimg.data;

  // copy
  for (int i = 0; i < inimg.rows * inimg.cols; ++i) {
    outimg_ptr[i].r = inimg_ptr[i * 3 + 0];
    outimg_ptr[i].g = inimg_ptr[i * 3 + 1];
    outimg_ptr[i].b = inimg_ptr[i * 3 + 2];
  }
}

// Converts from gSLICr image to OpenCV matrix
void SuperPixelFilter::convertImage(const gSLICr::UChar4Image* inimg,
                                    cv::Mat& outimg) {
  ROS_ASSERT(inimg);

  outimg = cv::Mat(inimg->noDims.y, inimg->noDims.x, CV_8UC3);
  unsigned char* outimg_ptr = outimg.data;
  const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

  // copy
  for (int i = 0; i < inimg->noDims.y * inimg->noDims.x; ++i) {
    outimg_ptr[i * 3 + 0] = inimg_ptr[i].r;
    outimg_ptr[i * 3 + 1] = inimg_ptr[i].g;
    outimg_ptr[i * 3 + 2] = inimg_ptr[i].b;
  }
}

// Overload for IntImage
int SuperPixelFilter::convertImage(const gSLICr::IntImage* inimg,
                                   cv::Mat& outimg, bool scaleForContrast) {
  ROS_ASSERT(inimg);

  // Read memory
  outimg = cv::Mat(inimg->noDims.y, inimg->noDims.x, CV_16UC1);
  unsigned short* outimg_ptr = (unsigned short*)outimg.data;
  const int* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

  // Find maximum
  int pixelCount = inimg->noDims.x * inimg->noDims.y;
  // Just get the value of the last pixel since we know the spixel order
  int spixels = inimg_ptr[pixelCount - 1] + 1;

  // Check that there aren't too many clusters to represent in gray format
  if (std::numeric_limits<unsigned short>::max() < spixels) {
    ROS_FATAL(
        "SuperPixelFilter: cannot produce mask because there are too many "
        "clusters: %i > %i",
        spixels, std::numeric_limits<unsigned short>::max());
    ROS_BREAK();
  }

  // copy
  for (int i = 0; i < inimg->noDims.y * inimg->noDims.x; ++i) {
    outimg_ptr[i] = inimg_ptr[i];
  }

  // Scale for contrast
  if (scaleForContrast) {
    // Find step size for visual contrast
    unsigned short step = std::numeric_limits<unsigned short>::max() / spixels;

    // Multiply entire matrix by step
    outimg *= step;
  }

  // Return number of superpixels
  return spixels;
}

void SuperPixelFilter::filterLabImage(const cv::Mat& input) {
  ROS_ASSERT(isInitialized());
  ROS_ASSERT(input.channels() == 3);

  // Resize input image if necessary
  cv::Mat resized;
  if (input.cols != colorImage_->noDims.x ||
      input.rows != colorImage_->noDims.y) {
    // Warn once per image resolution change
    if (resizeWarnWidth_ == -1 || resizeWarnWidth_ != input.cols ||
        resizeWarnHeight_ != input.rows) {
      ROS_WARN(
          "SuperPixelFilter: filtered image was resized to %i x %i from %i x "
          "%i (warn once per new resolution)",
          colorImage_->noDims.x, colorImage_->noDims.y, input.cols, input.rows);
      resizeWarnWidth_ = input.cols;
      resizeWarnHeight_ = input.rows;
    }

    // Use GPU to resize
    cv::cuda::GpuMat dev_input, dev_resized;
    dev_input.upload(input);
    cv::cuda::resize(dev_input, dev_resized,
                     cv::Size(colorImage_->noDims.x, colorImage_->noDims.y));
    dev_resized.download(resized);

    // Ensure matrix is continuous
    ROS_ASSERT(resized.isContinuous());
  } else {
    resized = input;
  }

  // Convert input to gSLICr image
  convertImage(resized, colorImage_.get());

  // Apply filter (will not change buffer contents)
  engine_->Process_Frame(colorImage_.get());

  initialFilterDone_ = true;
}

int SuperPixelFilter::resultMask(cv::Mat& out, bool scaleForContrast) {
  ROS_ASSERT(initialFilterDone_);
  const gSLICr::IntImage* grayImage = engine_->Get_Seg_Res();
  int spixels = convertImage(grayImage, out, scaleForContrast);
  return spixels;
}

void SuperPixelFilter::resultLineOverlay(cv::Mat& out) {
  ROS_ASSERT(initialFilterDone_);

  // Copy original image to overlay buffer so that we do not modify it
  gSLICr::Vector4u* overlayImageData = overlayImage_->GetData(MEMORYDEVICE_CPU);
  const gSLICr::Vector4u* colorImageData =
      colorImage_->GetData(MEMORYDEVICE_CPU);
  memcpy(
      overlayImageData, colorImageData,
      colorImage_->noDims.x * colorImage_->noDims.y * sizeof(gSLICr::Vector4u));

  // Draw and return
  engine_->Draw_Segmentation_Result(overlayImage_.get());
  convertImage(overlayImage_.get(), out);
}

void SuperPixelFilter::resultAverageColorMask(
    cv::Mat& colorMaskOut, const std::vector<cv::Scalar>& inColorList) {
  ROS_ASSERT(initialFilterDone_);

  // Read memory for mask
  // Note that it is assumed that mask values will range from 0 to
  // NumberOfSuperpixels
  const gSLICr::IntImage* grayImage = engine_->Get_Seg_Res();
  const int* grayImageData = grayImage->GetData(MEMORYDEVICE_CPU);

  // Find spixel count
  int pixelCount = grayImage->noDims.x * grayImage->noDims.y;
  // Just get the value of the last pixel since we know the spixel order
  int spixels = grayImageData[pixelCount - 1] + 1;

  // Verify that mask and buffer have same dimensions
  ROS_ASSERT(pixelCount == colorImage_->noDims.x * colorImage_->noDims.y);

  // Buffer metrics
  size_t spixelBinsBytes = sizeof(gSLICr::Vector4i) * spixels;
  size_t grayImageBufferBytes = sizeof(int) * pixelCount;

  // Allocate memory / assign textures for caching
  if (!initialColorAverageDone_) {
    // Allocations
    spixelBins_.reset(new gSLICr::Vector4i[spixels]);
    gpuErrorCheck(cudaMalloc(&dev_spixelBins_, spixelBinsBytes));
    gpuErrorCheck(cudaMalloc(&dev_grayImageData_, grayImageBufferBytes));

    // Gray image texture
    cudaResourceDesc grayResDesc;
    memset(&grayResDesc, 0, sizeof(grayResDesc));
    grayResDesc.resType = cudaResourceTypeLinear;
    grayResDesc.res.linear.devPtr = dev_grayImageData_;
    grayResDesc.res.linear.sizeInBytes = grayImageBufferBytes;

    cudaChannelFormatDesc grayChannelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    grayResDesc.res.linear.desc = grayChannelDesc;

    cudaTextureDesc grayTexDesc;
    memset(&grayTexDesc, 0, sizeof(grayTexDesc));
    grayTexDesc.readMode = cudaReadModeElementType;

    gpuErrorCheck(cudaCreateTextureObject(&grayTex_, &grayResDesc, &grayTexDesc,
                                          nullptr));

    // Color image texture
    cudaResourceDesc colorResDesc;
    memset(&colorResDesc, 0, sizeof(colorResDesc));
    colorResDesc.resType = cudaResourceTypeLinear;
    colorResDesc.res.linear.devPtr = dev_colorImageData_;
    colorResDesc.res.linear.sizeInBytes = colorImageBufferBytes_;

    cudaChannelFormatDesc colorChannelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    colorResDesc.res.linear.desc = colorChannelDesc;

    cudaTextureDesc colorTexDesc;
    memset(&colorTexDesc, 0, sizeof(colorTexDesc));
    colorTexDesc.readMode = cudaReadModeElementType;

    gpuErrorCheck(cudaCreateTextureObject(&colorTex_, &colorResDesc,
                                          &colorTexDesc, nullptr));
  }

  // for async CUDA
  cudaStream_t stream;
  gpuErrorCheck(cudaStreamCreate(&stream));

  // copy color list to change format
  // note that pixel colors must be ordered for the drawing kernel (third
  // component is gray id)
  auto sorted = inColorList;
  std::sort(sorted.begin(), sorted.end(),
            [](const cv::Scalar& a, const cv::Scalar& b) -> bool {
              return a[3] < b[3];
            });
  ROS_ASSERT(sorted.size() >= spixels);
  for (int i = 0; i < spixels; ++i) {
    spixelBins_[i].x = sorted[i][0];
    spixelBins_[i].y = sorted[i][1];
    spixelBins_[i].z = sorted[i][2];
  }

  // Copy the data
  gSLICr::Vector4u* colorImageData = colorImage_->GetData(MEMORYDEVICE_CPU);
  gpuErrorCheck(cudaMemcpyAsync(dev_spixelBins_, spixelBins_.get(),
                                spixelBinsBytes, cudaMemcpyHostToDevice,
                                stream));
  gpuErrorCheck(cudaMemcpyAsync(dev_grayImageData_, grayImageData,
                                grayImageBufferBytes, cudaMemcpyHostToDevice,
                                stream));
  gpuErrorCheck(cudaMemcpyAsync(dev_colorImageData_, colorImageData,
                                colorImageBufferBytes_, cudaMemcpyHostToDevice,
                                stream));
  gpuErrorCheck(cudaStreamSynchronize(stream));

  // Run drawing kernel
  // Note that we don't need to copy anything to the device for this call since
  // everything is already loaded
  cv::Mat colorMask(colorImage_->noDims.y, colorImage_->noDims.x, CV_8UC3);
  callDrawAverageColors_device(blocks_, threadsPerBlock_, dev_spixelBins_,
                               dev_openCvImageData_, colorImage_->noDims.y,
                               colorImage_->noDims.x, grayTex_);
  gpuErrorCheck(cudaMemcpyAsync(colorMask.data, dev_openCvImageData_,
                                openCvImageBufferBytes_, cudaMemcpyDeviceToHost,
                                stream));

  gpuErrorCheck(cudaStreamSynchronize(stream));
  gpuErrorCheck(cudaStreamDestroy(stream));

  // Return
  initialColorAverageDone_ = true;
  colorMaskOut = colorMask;
}

void SuperPixelFilter::resultColors(std::vector<cv::Scalar>& colorList) {
  ROS_ASSERT(initialFilterDone_);

  // fetch from gSLICr
  const gSLICr::SpixelMap* map = engine_->Get_Spixel_Map();

  // build list
  std::vector<cv::Scalar> vec;
  vec.reserve(map->noDims.x * map->noDims.y);
  const gSLICr::objects::spixel_info* map_ptr = map->GetData(MEMORYDEVICE_CPU);
  for (int i = 0; i < map->noDims.x * map->noDims.y; ++i) {
    vec.push_back(cv::Scalar(map_ptr[i].color_info[0], map_ptr[i].color_info[1],
                             map_ptr[i].color_info[2], map_ptr[i].id));
  }

  // return
  colorList = vec;
}

// Adjusts the region of interest if the spfilter and image are different sizes
void SuperPixelFilter::adjustRoi(au_core::Roi& roi,
                                 const cv::Mat& image) const {
  if (image.rows != height() && image.cols != width()) {
    roi.width = static_cast<unsigned int>(
        ((double)roi.width * (double)image.cols) / (double)width());
    roi.height = static_cast<unsigned int>(
        ((double)roi.height * (double)image.rows) / (double)height());

    roi.topLeft.x = (roi.topLeft.x * (double)image.cols) / (double)width();
    roi.topLeft.y = (roi.topLeft.y * (double)image.rows) / (double)height();

    // Adjust contour (downscale back to original image)
    for (int i = 0; i < roi.polygon.size(); ++i) {
      roi.polygon[i].x = (int)((double)roi.polygon[i].x *
                               ((double)image.cols / (double)width()));
      roi.polygon[i].y = (int)((double)roi.polygon[i].y *
                               ((double)image.rows / (double)height()));
    }
  }
}

}  // namespace au_vision
