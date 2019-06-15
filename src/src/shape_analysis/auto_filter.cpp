#include <au_vision/shape_analysis/auto_filter.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <limits>

namespace au_vision {

// Generates a graphical display for a 2d histogram
void imageLabHistogram(const cv::Mat& hist, cv::Mat& outputRgb, int blocksize,
                       int totalPixels, int minDisplayVal) {
  int lineThickness = 4;
  int imageHeight = hist.rows * blocksize;
  int imageWidth = hist.cols * blocksize;
  outputRgb = cv::Mat(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int y = 0; y < hist.rows; ++y) {
    for (int x = 0; x < hist.cols; ++x) {
      int brightness =
          (int)((double)hist.at<int>(y, x) / (double)totalPixels * 255.0);

      // LAB Color Guide
      cv::Mat pixels(
          1, 1, CV_8UC3,
          cv::Scalar(255, y * (255 / hist.rows), x * (255 / hist.cols)));
      cv::cvtColor(pixels, pixels, cv::COLOR_Lab2RGB);

      cv::Point a(x * blocksize, y * blocksize);
      cv::Point b(x * blocksize + blocksize, y * blocksize + blocksize);
      cv::rectangle(outputRgb, a, b,
                    cv::Scalar(brightness, brightness, brightness), CV_FILLED);
      cv::rectangle(outputRgb, a, b, pixels.at<cv::Vec3b>(0, 0), lineThickness);

      cv::Point n(x * blocksize + blocksize / 4, y * blocksize + blocksize / 4);
      cv::Point m(x * blocksize + blocksize / 2, y * blocksize + blocksize / 2);
      if (hist.at<int>(y, x) != 0 && hist.at<int>(y, x) < minDisplayVal) {
        cv::rectangle(outputRgb, n, m, cv::Scalar(100, 100, 255), CV_FILLED);
      } else if (hist.at<int>(y, x) == 0) {
        cv::rectangle(outputRgb, n, m, cv::Scalar(255, 0, 0), CV_FILLED);
      }

      // Draw frequency
      std::stringstream floatString;
      floatString << hist.at<int>(y, x);  // Use non-normal hist
      int tc;
      if (brightness > 200)
        tc = 0;
      else
        tc = 255;
      cv::putText(outputRgb, floatString.str().c_str(),
                  cv::Point(x * blocksize + 2, (y + 1) * blocksize - 5),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(tc, tc, tc));
    }
  }
}

void maskFromAutoFilter(const cv::Mat& imageLab,
                        const std::vector<cv::Scalar>& colorsIn,
                        int uniformBins, int blackMargin, int whiteMargin,
                        cv::Mat* histogramVisualOut, cv::Mat& maskOut) {
  ROS_ASSERT(imageLab.channels() == 3);

  // Build histogram
  cv::Mat hist(uniformBins, uniformBins, CV_32SC1, cv::Scalar(0));
  int squareSize = 255 / uniformBins;
  ROS_ASSERT(255 % uniformBins == 0);
  for (auto c : colorsIn) {
    hist.ptr<int>(static_cast<int>(c[1]) /
                  squareSize)[static_cast<int>(c[2]) / squareSize] += 1;
  }

  // Get debug image
  if (histogramVisualOut != nullptr) {
    imageLabHistogram(hist, *histogramVisualOut, 64, colorsIn.size(), 100);
  }

  // Search for clusters of non-zero histogram values
  std::vector<std::vector<cv::Point>> clusters;
  clusters = findClusters<int>(hist, 0);

  // Build value mapper (for mask creation in single pass)
  // Add two to make space for black and white
  int thisValue = 0;
  unsigned short* valueMap = new unsigned short[255 * 255];
  memset(valueMap, 0, 255 * 255 * sizeof(unsigned short));
  for (auto cluster : clusters) {
    for (auto color : cluster) {
      for (int y = color.y * squareSize;
           y < (color.y + 1) * squareSize && y < 255; ++y) {
        for (int x = color.x * squareSize;
             x < (color.x + 1) * squareSize && x < 255; ++x) {
          valueMap[y * 255 + x] = thisValue;
        }
      }
    }
    thisValue += 1;
  }

  // Build mask
  unsigned short* dev_valueMap;
  unsigned short* dev_maskBytes;
  unsigned char* dev_imageLab;
  cv::Mat mask(imageLab.rows, imageLab.cols, CV_16UC1);
  gpuErrorCheck(cudaMalloc(&dev_valueMap, 255 * 255 * sizeof(unsigned short)));
  gpuErrorCheck(cudaMalloc(&dev_maskBytes,
                           mask.rows * mask.cols * sizeof(unsigned short)));
  gpuErrorCheck(cudaMalloc(&dev_imageLab, imageLab.rows * imageLab.cols * 3));
  gpuErrorCheck(cudaMemcpy(dev_valueMap, valueMap,
                           255 * 255 * sizeof(unsigned short),
                           cudaMemcpyHostToDevice));
  gpuErrorCheck(cudaMemcpy(dev_imageLab, imageLab.data,
                           imageLab.rows * imageLab.cols * 3,
                           cudaMemcpyHostToDevice));

  callBuildMask_device(dev_maskBytes, dev_valueMap, dev_imageLab,
                       mask.rows * mask.cols);

  gpuErrorCheck(cudaMemcpy(mask.data, dev_maskBytes,
                           mask.rows * mask.cols * sizeof(unsigned short),
                           cudaMemcpyDeviceToHost));

  delete[] valueMap;
  gpuErrorCheck(cudaFree(dev_valueMap));
  gpuErrorCheck(cudaFree(dev_maskBytes));
  gpuErrorCheck(cudaFree(dev_imageLab));

  // Also use simple thresholding to get black and white
  cv::cuda::GpuMat dev_colorMask(imageLab), dev_blackMask, dev_whiteMask,
      dev_mask(mask);
  cudaStream_t stream;
  gpuErrorCheck(cudaStreamCreate(&stream));

  dev_blackMask.create(dev_colorMask.rows, dev_colorMask.cols, CV_8UC1);
  dev_whiteMask.create(dev_colorMask.rows, dev_colorMask.cols, CV_8UC1);

  ColorRange black(cv::Scalar(0, 128, 128), blackMargin);
  ColorRange white(cv::Scalar(255, 128, 128), whiteMargin);

  callInRange_device(dev_colorMask, black.lower, black.upper, dev_blackMask,
                     stream);
  callInRange_device(dev_colorMask, white.lower, white.upper, dev_whiteMask,
                     stream);

  gpuErrorCheck(cudaStreamSynchronize(stream));
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK
  gpuErrorCheck(cudaStreamDestroy(stream));

  // Add black and white to auto filter mask
  cv::cuda::Stream cvStream;
  dev_mask.setTo(thisValue, dev_blackMask, cvStream);
  dev_mask.setTo(thisValue + 1, dev_whiteMask, cvStream);
  cvStream.waitForCompletion();

  // Output
  dev_mask.download(mask);
  int scaledValueStep = std::numeric_limits<unsigned short>::max() / thisValue;
  maskOut = mask * scaledValueStep;
}

}  // namespace au_vision
