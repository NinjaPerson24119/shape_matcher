#ifdef GPU
#ifndef AU_VISION_AUTO_FILTER
#define AU_VISION_AUTO_FILTER

#include <list>
#include <vector>

#include <ros/ros.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

namespace au_vision {

// Generates a debug image from a histogram
void imageLabHistogram(const cv::Mat& hist, cv::Mat& outputRgb, int blocksize,
                       int totalPixels, int minDisplayVal);

// Uses the histogram technique to make a mask (for a single find contours
// function call)
void maskFromAutoFilter(const cv::Mat& imageLab,
                        const std::vector<cv::Scalar>& colorsIn,
                        int uniformBins, int blackMargin, int whiteMargin,
                        cv::Mat* histogramVisualOut, cv::Mat& maskOut);

/**
 * @brief Finds bin points adjacent to the given bin (including diagonal
 * touching squares) (not including bins with frequencies of ZeroValue)
 *
 * @templateparam T This is the type of object contained within the histogram
 * (float, byte, int, etc.)
 *
 * @param Histogram A 2d histogram to search
 * @param x the x coordinate of the bin to search at
 * @param y the y coordinate of the bin to search at
 * @param ZeroValue the object to use for comparisons. When the element of the
 * histogram is equal to this, said value is never added as adjacent to
 * anything. Any space equaling this is "empty".
 * @param IgnoreList a list of points to ignore during the search (Note: by
 * default, the point to search at is included in the return)
 *
 * @return the list of adjacent bin points
 *
 */
template <class T>
std::vector<cv::Point> findAdjacentIndices(
    const cv::Mat& Histogram, int x, int y, const T& ZeroValue,
    const std::vector<cv::Point>& IgnoreList) {
  std::vector<cv::Point> indicesOfAdjacents;
  indicesOfAdjacents.push_back(cv::Point(x, y));

  // Add indices
  if (Histogram.at<T>(y, x) > ZeroValue) {
    if (x - 1 >= 0)  // Left
    {
      if (Histogram.at<T>(y, x - 1) > ZeroValue) {
        indicesOfAdjacents.push_back(cv::Point(x - 1, y));
      }

      if (x + 1 < Histogram.cols)  // Right
      {
        if (Histogram.at<T>(y, x + 1) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x + 1, y));
        }
      }

      if (y - 1 >= 0)  // Up
      {
        if (Histogram.at<T>(y - 1, x) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x, y - 1));
        }
      }

      if (y + 1 < Histogram.rows)  // Down
      {
        if (Histogram.at<T>(y + 1, x) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x, y + 1));
        }
      }

      if ((x - 1 >= 0) && (y - 1 >= 0))  // Diagonal Up-Left
      {
        if (Histogram.at<T>(y - 1, x - 1) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x - 1, y - 1));
        }
      }

      if ((x + 1 < Histogram.cols) && (y - 1 >= 0))  // Diagonal Up-Right
      {
        if (Histogram.at<T>(y - 1, x + 1) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x + 1, y - 1));
        }
      }

      if ((x - 1 >= 0) && (y + 1 < Histogram.rows))  // Diagonal Down-Left
      {
        if (Histogram.at<T>(y + 1, x - 1) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x - 1, y + 1));
        }
      }

      if ((x + 1 < Histogram.cols) &&
          (y + 1 < Histogram.rows))  // Diagonal Down-Right
      {
        if (Histogram.at<T>(y + 1, x + 1) > ZeroValue) {
          indicesOfAdjacents.push_back(cv::Point(x + 1, y + 1));
        }
      }
    }
  }

  // Remove indices to be ignored
  for (int i = 0; i < IgnoreList.size(); ++i) {
    for (int j = 0; j < indicesOfAdjacents.size(); ++j) {
      // if in ignore list then flag for deletion
      if (indicesOfAdjacents[j] == IgnoreList[i]) {
        // assign delete flag
        indicesOfAdjacents[j].x = -1;
      }
    }
  }

  // Rebuild totalIndices
  std::vector<cv::Point> newIndicesOfAdjacents;
  newIndicesOfAdjacents.reserve(indicesOfAdjacents.size());
  for (auto i : indicesOfAdjacents) {
    // copy if not flagged
    if (i.x != -1) {
      newIndicesOfAdjacents.push_back(i);
    }
  }

  return newIndicesOfAdjacents;
}

/**
 * @brief Finds bin points connected to the given bin (including touching
 * corners) (not including bins with frequencies of ZeroValue)
 *
 * @templateparam T This is the type of object contained within the histogram
 * (float, byte, int, etc.)
 *
 * @param Histogram A 2d histogram to search
 * @param x the x coordinate of the point to search at
 * @param y the y coordinate of the point to search at
 * @param ZeroValue the object to use for comparisons. When the element of the
 * histogram is equal to this, said value is never added as adjacent to
 * anything. Any space equaling this is "empty".
 * @param IgnoreList a list of points to ignore during the search (Note: by
 * default, the the point to search at is included in the return)
 *
 * @return the list of connected bin points
 *
 */
template <class T>
std::vector<cv::Point> findAdjacentIndicesRecursive(
    const cv::Mat& Histogram, int x, int y, const T& ZeroValue,
    const std::vector<cv::Point>& IgnoreList) {
  std::vector<cv::Point> firstIgnore;
  firstIgnore.push_back(cv::Point(x, y));

  std::vector<cv::Point> totalIndices;  // Will also be used as ignore list
  cv::Point thisPoint(x, y);

  // Add ignore indices
  if (Histogram.at<T>(y, x) > ZeroValue)  // Do not add 0 freq elements
  {
    totalIndices.push_back(thisPoint);  // Do not forget original indice
  } else {
    // don't start searches from zero squares
    return std::vector<cv::Point>();
  }

  std::vector<cv::Point> toSearch = findAdjacentIndices<T>(
      Histogram, x, y, ZeroValue, firstIgnore);  // Initial search
  for (auto i : toSearch) {
    totalIndices.push_back(i);
  }

  // Keep searching until all surrounding elements d.n.e. or are 0
  while (toSearch.size() != 0) {
    // Get the last of the total indices that was be searched
    int nextSearch = totalIndices.size();

    // Search
    for (int k = 0; k < toSearch.size(); ++k) {
      std::vector<cv::Point> newSearch = findAdjacentIndices<T>(
          Histogram, toSearch[k].x, toSearch[k].y, ZeroValue, totalIndices);

      // Add new elements to total list
      for (auto i : newSearch) {
        totalIndices.push_back(i);
      }
    }

    // Clear searched elements
    toSearch.clear();

    // Add new elements to be searched
    for (int i = nextSearch; i < totalIndices.size(); ++i) {
      toSearch.push_back(totalIndices[i]);
    }
  }

  // Remove indices to be ignored
  for (int i = 0; i < IgnoreList.size(); ++i) {
    for (int j = 0; j < totalIndices.size(); ++j) {
      // if in ignore list then flag for deletion
      if (totalIndices[j] == IgnoreList[i]) {
        // assign delete flag
        totalIndices[j].x = -1;
      }
    }
  }

  // Rebuild totalIndices
  std::vector<cv::Point> newTotalIndices;
  newTotalIndices.reserve(totalIndices.size());
  for (auto i : totalIndices) {
    // copy if not flagged
    if (i.x != -1) {
      newTotalIndices.push_back(i);
    }
  }

  return newTotalIndices;
}

/**
 * @brief Finds all disconnected clusters of bins (not including bins with
 * frequencies of ZeroValue)
 * @param Histogram A 2d histogram to search
 * @param ZeroValue the object to use for comparisons. When the element of the
 * histogram is equal to this, said value is never added as adjacent to
 * anything. Any space equaling this is "empty".
 *
 * @return the vector of point lists (clusters)
 *
 */
template <class T>
std::vector<std::vector<cv::Point> > findClusters(const cv::Mat& Histogram,
                                                  const T& ZeroValue) {
  std::vector<cv::Point> knownIndices;
  std::vector<std::vector<cv::Point> > clusters;
  for (int y = 0; y < Histogram.rows; ++y) {
    for (int x = 0; x < Histogram.cols; ++x) {
      // Look for cluster that does not have known indices
      std::vector<cv::Point> temp = findAdjacentIndicesRecursive<int>(
          Histogram, x, y, ZeroValue, knownIndices);

      if (temp.size() > 0) {
        clusters.push_back(temp);
      }

      // Update known indices
      for (auto i : temp) {
        knownIndices.push_back(i);
      }
    }
  }
  return clusters;
}

}  // namespace au_vision

#endif
#endif
