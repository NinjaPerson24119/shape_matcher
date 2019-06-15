#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <au_vision/shape_analysis/shape_analysis_matcher.h>

#include <ros/ros.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

// DEBUG TODO
#include <ctime>

namespace au_vision {

double groupAverageAreaRatio(
    const std::vector<std::vector<cv::Point>>& realtimeContours,
    const ShapeDb& db, const ShapeGroupMatch& match) {
  // Get average RT contour area
  double averageAreaRatio = 0;
  const ShapeGroup& group = db.groups[match.groupIndex];
  for (int i = 0; i < match.parts.size(); ++i) {
    double thisRtArea =
        cv::contourArea(realtimeContours[match.parts[i].rtIndex]);
    double thisDbArea = db.shapes[match.parts[i].dbIndex].area;
    averageAreaRatio += thisDbArea / thisRtArea;
  }
  averageAreaRatio /= match.parts.size();

  return averageAreaRatio;
}

double approximateGroupDistance(double groupAreaRatio,
                                const au_core::CameraInfo& cameraInfo) {
  /*
  // Get average RT contour area
  double averageAreaRatio = 0;
  ShapeGroup& group = db.groups[match.groupIndex];
  for (int i = 0; i < match.parts.size(); ++i) {
    double thisRtArea =
        cv::contourArea(realtimeContours[match.parts[i].rtIndex]);
    double thisDbArea = db.shapes[group.shapeIndices[i]].area;
    averageAreaRatio += thisDbArea / thisRtArea;
  }
  averageAreaRatio /= match.parts.size();

  // Scale according to render distance
  return averageAreaRatio * db.renderDistance;
  */
  return 0;  // DEBUG because SEGFAULT
}

cv::Rect scaledGroupBound(const ShapeDb& db,
                          const ShapeGroupMatch& groupMatch) {
  ROS_ASSERT(groupMatch.parts.size());

  cv::Rect finalBound;
  finalBound.width = 0;
  finalBound.height = 0;
  finalBound.x = 0;
  finalBound.y = 0;

  for (int i = 0; i < groupMatch.parts.size(); ++i) {
    // Pick first shape
    const ShapeMatch& match = groupMatch.parts[i];
    const Shape& shape = db.shapes[match.dbIndex];
    const ShapeGroup& group = db.groups[groupMatch.groupIndex];

    // Get dim ratios
    double xRatio = (double)match.bound.width / (double)shape.bound.width;
    double yRatio = (double)match.bound.height / (double)shape.bound.height;

    // Scale to get group bound
    finalBound.width += (double)group.bound.width * xRatio;
    finalBound.height += (double)group.bound.height * yRatio;
    finalBound.x += match.bound.x - shape.bound.x * xRatio;
    finalBound.y += match.bound.y - shape.bound.y * yRatio;

    if (i != 0) {
      finalBound.width /= 2;
      finalBound.height /= 2;
      finalBound.x /= 2;
      finalBound.y /= 2;
    }
  }

  // Return
  return finalBound;
}

std::vector<ShapeGroupMatch> matchRealtimeContours(
    std::vector<std::vector<cv::Point>>& realtimeContours,
    std::vector<ShapeColor>& realtimeColors, ShapeDb& db,
    MatchShapesThresholds& shapeTh, MatchShapeGroupThresholds& groupTh,
    std::vector<double>& bestLooseRatings,
    const au_core::CameraInfo& cameraInfo, int parallelFrames) {
  ROS_ASSERT(parallelFrames);
  ROS_ASSERT(realtimeColors.size() == realtimeContours.size());

  // return immediately if there's no work to be done
  if (realtimeContours.size() == 0) {
    return std::vector<ShapeGroupMatch>();
  }

  size_t roughAreaDifferencesBytes = db.shapes.size() * sizeof(int);
  size_t colorDifferencesBytes = db.shapes.size() * sizeof(int);
  size_t compressedImageBytes =
      (db.frameBufferWidth * db.frameBufferHeight) / 8;

  // Allocations
  std::vector<int*> dev_roughAreaDifferences(realtimeContours.size());
  std::vector<int*> dev_interiorColorDifferences(realtimeContours.size());
  std::vector<int*> dev_exteriorColorDifferences(realtimeContours.size());
  std::vector<int*> roughAreaDifferences(realtimeContours.size());
  std::vector<int*> interiorColorDifferences(realtimeContours.size());
  std::vector<int*> exteriorColorDifferences(realtimeContours.size());
  uint8_t* rtImage;
  uint32_t* dev_dbImages;
  uint32_t* dev_dbImagesResult;
  int* dev_areaDifferences;
  int* areaDifferences;
  std::vector<uint32_t*> dev_rtImages(realtimeContours.size());

  for (int r = 0; r < realtimeContours.size(); ++r) {
    gpuErrorCheck(
        cudaMalloc(&dev_roughAreaDifferences[r], roughAreaDifferencesBytes));
    gpuErrorCheck(
        cudaMalloc(&dev_interiorColorDifferences[r], colorDifferencesBytes));
    gpuErrorCheck(
        cudaMalloc(&dev_exteriorColorDifferences[r], colorDifferencesBytes));
    roughAreaDifferences[r] = new int[db.shapes.size()];
    interiorColorDifferences[r] = new int[db.shapes.size()];
    exteriorColorDifferences[r] = new int[db.shapes.size()];

    gpuErrorCheck(cudaMalloc(&dev_rtImages[r], compressedImageBytes));
  }
  rtImage = new uint8_t[compressedImageBytes];

  gpuErrorCheck(
      cudaMalloc(&dev_dbImages, compressedImageBytes * parallelFrames));
  gpuErrorCheck(
      cudaMalloc(&dev_dbImagesResult, compressedImageBytes * parallelFrames));

  gpuErrorCheck(cudaMalloc(&dev_areaDifferences, sizeof(int) * parallelFrames));
  areaDifferences = new int[parallelFrames];

  // for async CUDA
  cudaStream_t stream;
  gpuErrorCheck(cudaStreamCreate(&stream));

  // rtMatchBins holds the potential matches for each shape in the DB
  std::vector<std::vector<ShapeMatch>> rtMatchBins(db.shapes.size());

  // holds the loose matches
  std::vector<std::vector<ShapeMatch>> rtMatches(realtimeContours.size());

  // prepare realtime contours for comparisons
  std::vector<ShapeDb> rtDb(realtimeContours.size(), ShapeDb(db));

  for (int r = 0; r < realtimeContours.size(); ++r) {
    // process the contour
    std::vector<cv::Point> processedContour;
    processedContour = realtimeContours[r];
    centerAndMaximizeContour(processedContour, db.frameBufferWidth,
                             db.frameBufferHeight);

    // build rt DB (make dummy group with single contour)
    Shape rtShape;
    rtShape.contour = processedContour;
    rtShape.name = "rtPart";
    rtShape.color = realtimeColors[r];
    std::vector<Shape> rtShapeVec;
    rtShapeVec.push_back(rtShape);
    cv::Rect tempBound;
    rtDb[r].addGroup(rtShapeVec, "rtGroup", cv::Scalar(0, 0, 0), tempBound);
    rtDb[r].loadToGpu();

    // Compare RT grids to DB grids (rough area difference)
    callRoughShapeAreaDifferences_device(
        rtDb[r].dev_grids, db.dev_grids, db.gridRows, db.gridCols,
        db.shapes.size(), dev_roughAreaDifferences[r], stream);

    // Compare RT and DB colors (color differences)
    callColorDifferences_device(
        rtDb[r].dev_interiorColors, db.dev_interiorColors,
        dev_interiorColorDifferences[r], db.shapes.size(), shapeTh.blackMargin,
        shapeTh.whiteMargin, stream);
    callColorDifferences_device(
        rtDb[r].dev_exteriorColors, db.dev_exteriorColors,
        dev_exteriorColorDifferences[r], db.shapes.size(), shapeTh.blackMargin,
        shapeTh.whiteMargin, stream);

    // unpack rt image to device
    packedToCompressed(rtDb[r].packedImages[0], rtImage);
    gpuErrorCheck(cudaMemcpy(dev_rtImages[r], rtImage, compressedImageBytes,
                             cudaMemcpyHostToDevice));
  }
  gpuErrorCheck(cudaStreamSynchronize(stream));
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

  // copy color differences back to host
  for (int r = 0; r < realtimeContours.size(); ++r) {
    gpuErrorCheck(cudaMemcpyAsync(
        roughAreaDifferences[r], dev_roughAreaDifferences[r],
        roughAreaDifferencesBytes, cudaMemcpyDeviceToHost, stream));
    gpuErrorCheck(cudaMemcpyAsync(
        interiorColorDifferences[r], dev_interiorColorDifferences[r],
        colorDifferencesBytes, cudaMemcpyDeviceToHost, stream));
    gpuErrorCheck(cudaMemcpyAsync(
        exteriorColorDifferences[r], dev_exteriorColorDifferences[r],
        colorDifferencesBytes, cudaMemcpyDeviceToHost, stream));
  }
  gpuErrorCheck(cudaStreamSynchronize(stream));
  gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

  // do area comparisons in batches of frames (so we only need to unpack each
  // image once)
  for (int k = 0; k < db.packedImages.size(); k += parallelFrames) {
    int batchSize = k + parallelFrames < db.packedImages.size()
                        ? parallelFrames
                        : db.packedImages.size() - k;

    // unpack one batch of images
    callPackedToCompressed_device(db.dev_packedImages + k * db.maxPacks,
                                  dev_dbImages, batchSize, compressedImageBytes,
                                  db.maxPacks, stream);
    gpuErrorCheck(cudaStreamSynchronize(stream));
    gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

    // compare this batch to each rt contour
    for (int r = 0; r < realtimeContours.size(); ++r) {
      // build list of shapes that weren't culled by the rough area differences
      std::vector<int> validIndices;
      for (int i = 0; i < batchSize; ++i) {
        if (roughAreaDifferences[r][k + i] < shapeTh.areaDifferenceThresh) {
          validIndices.push_back(i);
        }
      }

      size_t uint32ImageStep = compressedImageBytes / sizeof(uint32_t);

      // XOR
      for (int n = 0; n < validIndices.size(); ++n) {
        uint32_t* dbImg = dev_dbImages + validIndices[n] * uint32ImageStep;
        uint32_t* dbImgResult =
            dev_dbImagesResult + validIndices[n] * uint32ImageStep;
        callCompressedXOR_device(dev_rtImages[r], dbImg, dbImgResult,
                                 uint32ImageStep, stream);
      }
      gpuErrorCheck(cudaStreamSynchronize(stream));
      gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

      // population count
      for (int n = 0; n < validIndices.size(); ++n) {
        uint32_t* dbImgResult =
            dev_dbImagesResult + validIndices[n] * uint32ImageStep;
        callPopulationCount_device(dbImgResult, dbImgResult, uint32ImageStep,
                                   stream);
      }
      gpuErrorCheck(cudaStreamSynchronize(stream));
      gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

      // sum
      gpuErrorCheck(
          cudaMemset(dev_areaDifferences, 0, batchSize * sizeof(int)));
      for (int n = 0; n < validIndices.size(); ++n) {
        uint32_t* dbImgResult =
            dev_dbImagesResult + validIndices[n] * uint32ImageStep;
        callSumImage_device(dbImgResult, dev_areaDifferences + validIndices[n],
                            uint32ImageStep, stream);
      }
      gpuErrorCheck(cudaStreamSynchronize(stream));
      gpuErrorCheck(cudaGetLastError());  // Verify that all went OK

      // copy threshold results and area differences back to host
      gpuErrorCheck(cudaMemcpy(areaDifferences, dev_areaDifferences,
                               batchSize * sizeof(int),
                               cudaMemcpyDeviceToHost));

      // identify shapes that pass thresholds
      for (int n = 0; n < validIndices.size(); ++n) {
        int idx = k + validIndices[n];
        bool passInteriorColor =
            (interiorColorDifferences[r][idx] < shapeTh.colorDifferenceThresh &&
             db.shapes[idx].color.validInterior &&
             realtimeColors[r].validInterior) ||
            !db.shapes[idx].color.validInterior;
        bool passExteriorColor =
            (exteriorColorDifferences[r][idx] < shapeTh.colorDifferenceThresh &&
             db.shapes[idx].color.validExterior &&
             realtimeColors[r].validExterior) ||
            !db.shapes[idx].color.validExterior;
        bool passAreaDifference =
            areaDifferences[validIndices[n]] < shapeTh.areaDifferenceThresh;

        if (passInteriorColor && passExteriorColor && passAreaDifference) {
          ShapeMatch temp;
          temp.rtIndex = r;
          temp.dbIndex = idx;
          temp.rating = 1 - (double)areaDifferences[validIndices[n]] /
                                (double)shapeTh.areaDifferenceThresh;
          rtMatchBins[idx].push_back(temp);
          rtMatches[r].push_back(temp);
        }
      }
    }
  }

  // deallocations
  for (int r = 0; r < realtimeContours.size(); ++r) {
    gpuErrorCheck(cudaFree(dev_roughAreaDifferences[r]));
    gpuErrorCheck(cudaFree(dev_interiorColorDifferences[r]));
    gpuErrorCheck(cudaFree(dev_exteriorColorDifferences[r]));
    delete[] roughAreaDifferences[r];
    delete[] interiorColorDifferences[r];
    delete[] exteriorColorDifferences[r];

    gpuErrorCheck(cudaFree(dev_rtImages[r]));
  }
  delete[] rtImage;

  gpuErrorCheck(cudaFree(dev_dbImages));
  gpuErrorCheck(cudaFree(dev_dbImagesResult));

  gpuErrorCheck(cudaFree(dev_areaDifferences));
  delete[] areaDifferences;

  gpuErrorCheck(cudaStreamDestroy(stream));

  // Note that this next section requires that *all* of the calculations for
  // *all* shapes have been completed. This next section is a reduction step so
  // will be done sequentially

  // Find the best match rating for each RT contour
  bestLooseRatings = std::vector<double>(realtimeContours.size(), 0.0);
  for (int i = 0; i < rtMatches.size(); ++i) {
    for (int j = 0; j < rtMatches[i].size(); ++j) {
      if (rtMatches[i][j].rating > bestLooseRatings[rtMatches[i][j].rtIndex]) {
        bestLooseRatings[rtMatches[i][j].rtIndex] = rtMatches[i][j].rating;
      }
    }
  }

  // Find the best match rating for each shape
  // Each bin is reduced to a single item
  std::vector<ShapeMatch> rtBestMatches(rtMatchBins.size());
  for (int i = 0; i < rtMatchBins.size(); ++i) {
    int thisBest = 0;
    int thisBestIndice = -1;
    for (int j = 0; j < rtMatchBins[i].size(); ++j) {
      if (rtMatchBins[i][j].rating > thisBest) {
        thisBest = rtMatchBins[i][j].rating;
        thisBestIndice = j;
      }
    }
    if (thisBestIndice != -1) {
      rtBestMatches[i] = rtMatchBins[i][thisBestIndice];
    }
  }

  // Cull groups
  std::vector<int> groupsToTry;
  for (int i = 0; i < db.groups.size(); ++i) {
    // Ensure there are enough found DB shapes to complete a minimum rating
    bool enoughNecessaryShapes = false;
    int hits = 0;
    int minHits = std::ceil(groupTh.minimumRating *
                            (double)db.groups[i].shapeIndices.size());

    if (minHits == 0) {
      minHits = 1;
    }

    for (auto j : db.groups[i].shapeIndices) {
      if (rtMatchBins[j].size()) {
        ++hits;
      }
    }
    if (hits >= minHits) {
      enoughNecessaryShapes = true;
    }

    // Ensure the required DB shapes map to different RT contours
    bool differentContours = true;
    if (enoughNecessaryShapes) {
      std::set<int> indices;
      for (auto j : db.groups[i].shapeIndices) {
        if (rtMatchBins[j].size()) {
          auto exists = indices.find(rtBestMatches[j].rtIndex);
          if (exists == indices.end()) {
            indices.insert(rtBestMatches[j].rtIndex);
          } else {
            differentContours = false;
            break;
          }
        }
      }
    }

    if (enoughNecessaryShapes && differentContours) {
      groupsToTry.push_back(i);
    }
  }

  // Match groups
  std::vector<double> groupRatings;
  for (auto g : groupsToTry) {
    double rating = 0;
    for (int k = 0; k < db.groups[g].shapeIndices.size(); ++k) {
      auto j = db.groups[g].shapeIndices[k];

      // Not all ratings must exist, so check
      // Though it is expected that post-cull groups have at least one rating
      if (!rtMatchBins[j].size()) {
        continue;
      }

      // Cull groups that do not satisfy the (shapes / group size) ratio thresh
      if (static_cast<double>(rtMatchBins[j].size()) /
              static_cast<double>(db.groups[g].shapeIndices.size()) <
          groupTh.minimumShapeGroupRatio) {
        rating = 0;
        break;
      }

      rating += rtBestMatches[j].rating;

      // Do final average
      if (k == db.groups[g].shapeIndices.size() - 1) {
        rating /= db.groups[g].shapeIndices.size();
      }
    }
    groupRatings.push_back(rating);
  }

  // Find best rated group
  double bestGroupRating = 0;
  int bestGroupIdx = -1;
  for (int g = 0; g < groupRatings.size(); ++g) {
    if (groupRatings[g] > bestGroupRating) {
      bestGroupRating = groupRatings[g];
      bestGroupIdx = g;
    }
  }

  // Build output group (single) list
  std::vector<ShapeGroupMatch> outGroups;
  if (bestGroupIdx != -1 && bestGroupRating > groupTh.minimumRating) {
    ShapeGroupMatch tempGroup;

    // Add parts
    for (auto j : db.groups[groupsToTry[bestGroupIdx]].shapeIndices) {
      // Only output contours that were matched
      if (rtMatchBins[j].size()) {
        tempGroup.parts.push_back(rtBestMatches[j]);
        tempGroup.parts[tempGroup.parts.size() - 1].bound =
            cv::boundingRect(realtimeContours[rtBestMatches[j].rtIndex]);
      }
    }

    tempGroup.groupIndex = groupsToTry[bestGroupIdx];
    tempGroup.rating = bestGroupRating;
    tempGroup.bound = scaledGroupBound(db, tempGroup);
    tempGroup.distance = approximateGroupDistance(0, cameraInfo);
    outGroups.emplace_back(tempGroup);
  }

  // Return
  return outGroups;
}

}  // namespace au_vision
