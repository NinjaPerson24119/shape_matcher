/**
 * @author Nicholas Wengel
 */ 

#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis_kernels.h>
#include <au_vision/shape_analysis/shape_analysis_matcher.h>
#include <ros/ros.h>
#include <algorithm>
#include <cmath>
#include <set>

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

std::vector<ShapeGroupMatch> matchRealtimeContours(
    std::vector<std::vector<cv::Point>>& realtimeContours,
    std::vector<cv::Scalar>& realtimeColors, ShapeDb& db,
    MatchShapesThresholds& shapeTh, MatchShapeGroupThresholds& groupTh,
    ContourRenderer& renderer, std::vector<double>& bestLooseRatings,
    const au_core::CameraInfo& cameraInfo) {
  int debugPreMatch = 0;
  int debugMatches = 0;

  // Verify grids and renderer and shapes list
  ROS_ASSERT(db.goodForDetect());
  ROS_ASSERT(db.rendererIsCompatible(renderer));

  // Allocations
  int* dev_areaDifferences;
  size_t areaDifferencesBytes = db.shapes.size() * sizeof(int);
  int* areaDifferences = new int[db.shapes.size()];
  gpuErrorCheck(cudaMalloc(&dev_areaDifferences, areaDifferencesBytes));

  int* dev_colorDifferences;
  size_t colorDifferencesBytes = db.shapes.size() * sizeof(int);
  int* colorDifferences = new int[db.shapes.size()];
  gpuErrorCheck(cudaMalloc(&dev_colorDifferences, colorDifferencesBytes));

  // Iterate over real time contours
  std::vector<std::vector<ShapeMatch>> rtMatchBins(db.shapes.size());
  std::vector<std::vector<ShapeMatch>> rtMatches(realtimeContours.size());

  for (int rtIndex = 0; rtIndex < realtimeContours.size(); ++rtIndex) {
    // Process the contour
    std::vector<cv::Point> processedContour = realtimeContours[rtIndex];
    centerAndMaximizeContour(processedContour, renderer.width(),
                             renderer.height());

    // Get RT DB (make dummy group with single contour)
    ShapeDb rtDb(db);
    Shape rtShape;
    rtShape.contour = processedContour;
    rtShape.name = "rtPart";
    rtShape.color = realtimeColors[rtIndex];
    std::vector<Shape> rtShapeVec;
    rtShapeVec.emplace_back(rtShape);
    cv::Rect tempBound;
    rtDb.addGroup(rtShapeVec, renderer, "rtGroup", cv::Scalar(0, 0, 0),
                  tempBound);
    rtDb.loadToGpu();

    // Compare RT grids to DB grids (rough area difference)
    callRoughShapeAreaDifferences_device(
        rtDb.dev_grids, db.dev_grids, db.gridRows, db.gridCols,
        db.shapes.size(), dev_areaDifferences, rtDb.dev_colors, db.dev_colors,
        dev_colorDifferences);

    // Load differences back to host
    gpuErrorCheck(cudaMemcpy(areaDifferences, dev_areaDifferences,
                             areaDifferencesBytes, cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(colorDifferences, dev_colorDifferences,
                             colorDifferencesBytes, cudaMemcpyDeviceToHost));

    // Cull based on shape thresholds
    std::vector<int> bestIndices;
    for (int i = 0; i < db.shapes.size(); ++i) {
      if (areaDifferences[i] < shapeTh.areaDifferenceThresh &&
          colorDifferences[i] < shapeTh.colorDifferenceThresh) {
        ++debugPreMatch;
        bestIndices.push_back(i);
      }
    }

    // Compare RT raster and DB raster (fine area difference)
    // Double check shape thresholds
    cv::Mat rtRaster, dbRaster;
    for (auto m : bestIndices) {
      // Render
      renderer.render(processedContour, rtRaster);
      renderer.render(db.shapes[m].contour, dbRaster);

      // Difference
      cv::bitwise_not(dbRaster, dbRaster);
      cv::bitwise_and(rtRaster, dbRaster, rtRaster);
      int thisDiff = cv::countNonZero(rtRaster);

      if (thisDiff < shapeTh.areaDifferenceThresh) {
        ShapeMatch temp;
        temp.rtIndex = rtIndex;
        temp.dbIndex = m;
        temp.rating =
            1.0 - (double)thisDiff / (double)shapeTh.areaDifferenceThresh;
        rtMatchBins[m].push_back(temp);
        rtMatches[rtIndex].push_back(temp);
        ++debugMatches;
      }
    }
  }

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
    for (auto j : db.groups[g].shapeIndices) {
      // Not all ratings must exist, so check
      // Though it is expected that post-cull groups have at least one rating
      if (rtMatchBins[j].size()) {
        rating += rtBestMatches[j].rating;
      }
    }
    rating /= db.groups[g].shapeIndices.size();
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

  // Build output group list
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

  // Deallocate
  gpuErrorCheck(cudaFree(dev_areaDifferences));
  gpuErrorCheck(cudaFree(dev_colorDifferences));
  delete[] areaDifferences;
  delete[] colorDifferences;

  // Stats
  // ROS_INFO("------------------");
  // ROS_INFO("Matches With Grid Cull: %i", debugPreMatch);
  // ROS_INFO("Matches With Fine Cull: %i", debugMatches);

  // Return
  return outGroups;
}

}  // namespace au_vision
