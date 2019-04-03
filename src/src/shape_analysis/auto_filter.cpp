/**
 * @author Nicholas Wengel
 */ 

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
  for (auto c : colorsIn) {
    hist.at<int>(c[1] / squareSize, c[2] / squareSize) += 1;
  }

  // Get debug image
  if (histogramVisualOut != nullptr) {
    imageLabHistogram(hist, *histogramVisualOut, 64, colorsIn.size(), 100);
  }

  // Search for clusters of non-zero histogram values
  std::list<std::list<cv::Point>> clusters;
  clusters = findClusters<int>(hist, 0);

  // Build value mapper (for mask creation in single pass)
  // Add two to make space for black and white
  int thisValue = 0;
  std::vector<std::vector<unsigned short>> valueMap(
      255, std::vector<unsigned short>(255, 0));
  for (auto cluster : clusters) {
    for (auto color : cluster) {
      for (int y = color.y * squareSize;
           y < (color.y + 1) * squareSize && y < 255; ++y) {
        for (int x = color.x * squareSize;
             x < (color.x + 1) * squareSize && x < 255; ++x) {
          valueMap[y][x] = thisValue;
        }
      }
    }
    thisValue += 1;
  }

  // Build mask
  cv::Mat mask(imageLab.rows, imageLab.cols, CV_16UC1);
  for (int y = 0; y < mask.rows; ++y) {
    for (int x = 0; x < mask.cols; ++x) {
      cv::Vec3b thisPixel = imageLab.at<cv::Vec3b>(y, x);
      mask.at<unsigned short>(y, x) = valueMap[thisPixel[1]][thisPixel[2]];
    }
  }

  // Also use simple thresholding to get black and white
  cv::cuda::GpuMat dev_colorMask(imageLab), dev_blackMask, dev_whiteMask;
  ColorRange black(cv::Scalar(0, 128, 128), blackMargin);
  ColorRange white(cv::Scalar(255, 128, 128), whiteMargin);
  callInRange_device(dev_colorMask, black.lower, black.upper, dev_blackMask);
  callInRange_device(dev_colorMask, white.lower, white.upper, dev_whiteMask);
  cv::Mat whiteMask, blackMask;
  dev_blackMask.download(blackMask);
  dev_whiteMask.download(whiteMask);

  // Add black and white to auto filter mask
  mask.setTo(thisValue, blackMask);
  mask.setTo(thisValue + 1, whiteMask);
  cv::Mat shadeMask(blackMask.rows, blackMask.cols, CV_8UC1,
                    cv::Scalar(128, 128, 128));
  shadeMask.setTo(0, blackMask);
  shadeMask.setTo(255, whiteMask);

  // Output
  int scaledValueStep = std::numeric_limits<unsigned short>::max() / thisValue;
  maskOut = mask * scaledValueStep;
}

}  // namespace au_vision
