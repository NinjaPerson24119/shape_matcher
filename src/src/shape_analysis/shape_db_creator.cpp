/**
 * @author Nicholas Wengel
 */ 

#include <au_core/sigint_handler.h>

#include <au_vision/shape_analysis/contour_renderer.h>
#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <au_vision/shape_analysis/superpixel_filter.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <fstream>
#include <limits>
#include <sstream>

#include <osg/Camera>
#include <osg/Image>
#include <osg/PositionAttitudeTransform>
#include <osg/Quat>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>

using namespace au_vision;

const double degToRad = osg::PI / 180;

// Predeclarations
void rotate(osg::ref_ptr<osg::PositionAttitudeTransform>& transform,
            double xRot, double yRot, double zRot);

void writeTextAndClearSs(std::stringstream& text, cv::Mat& image,
                         cv::Point& where, int lineHeight);

std::vector<ColorRange> getFiltersFromParams(ros::NodeHandle& private_nh);

void validateFilters(std::vector<ColorRange>& filters);

int calcShapesFromImage(const cv::Mat& imageRgb,
                        const std::vector<ColorRange>& filters,
                        SuperPixelFilter& spFilter, ContourRenderer& renderer,
                        int rotX, int rotY, int rotZ, const std::string& name,
                        cv::Mat& viewOut, cv::Mat& viewOutProcessed,
                        bool gSLICrOn, ShapeDb& db, int minimumContourArea,
                        double contourLinearizationEpsilon);

// Entry point
int main(int argc, char** argv) {
  // Private node handle
  au_core::handleInterrupts(argc, argv, "shape_db_creator", true);
  ros::NodeHandle private_nh("~");

  // Set up ROS out
  image_transport::ImageTransport imageTransport(private_nh);
  image_transport::Publisher renderedPub =
      imageTransport.advertise("transformView", 10);
  image_transport::Publisher contourPub =
      imageTransport.advertise("contourView", 10);
  image_transport::Publisher contourProcessedPub =
      imageTransport.advertise("contourProcessedView", 10);

  // Load grids / FBO info (this is read first since OpenSceneGraph depends on
  // it)
  int textureWidth, textureHeight, gridSquareSize;
  if (!private_nh.getParam("FrameBufferWidth", textureWidth) ||
      !private_nh.getParam("FrameBufferHeight", textureHeight) ||
      !private_nh.getParam("SquareSize", gridSquareSize)) {
    ROS_FATAL("Missing grid / FBO param");
    ROS_BREAK();
  }

  // Load clear color
  cv::Scalar background;
  if (!private_nh.getParam("BackgroundR", background[0]) ||
      !private_nh.getParam("BackgroundG", background[1]) ||
      !private_nh.getParam("BackgroundB", background[2])) {
    ROS_FATAL("Missing grid / FBO param");
    ROS_BREAK();
  }
  for (int i = 0; i < 3; ++i) {
    background[i] /= 255;
  }

  // OpenSceneGraph help sources
  // Render to texture help source:
  // http://beefdev.blogspot.ca/2012/01/render-to-texture-in-openscenegraph.html
  // OpenSceneGraph 3.0 Beginners Guide, specifically the section at Pg: 183

  // Do anti aliasing
  osg::DisplaySettings::instance()->setNumMultiSamples(4);

  // Allocate image, which we will render to, then access on client
  osg::ref_ptr<osg::Image> image = new osg::Image;
  image->allocateImage(textureWidth, textureWidth, 1, GL_RGB, GL_UNSIGNED_BYTE);

  // Create the camera object
  osg::ref_ptr<osg::Camera> camera = new osg::Camera();

  // Render using frame buffer for speed
  camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);

  // Bind the image to the color buffer.
  // E.g. when the color buffer is updated, so is the image
  camera->attach(osg::Camera::COLOR_BUFFER, image.get());

  // Set absolute reference frame
  camera->setReferenceFrame(osg::Camera::ABSOLUTE_RF);

  // Render prior to sending to viewer
  camera->setRenderOrder(osg::Camera::PRE_RENDER);

  // Set viewport to texture size
  camera->setViewport(0, 0, textureWidth, textureHeight);

  // Set clear color and buffers to be cleared
  camera->setClearColor(
      osg::Vec4(background[0], background[1], background[2], 0.0f));
  camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Load model
  std::string modelString;
  if (!private_nh.getParam("model", modelString)) {
    ROS_FATAL("Model path was not passed");
    return -1;
  }
  osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(modelString);
  if (!model.get()) {
    ROS_FATAL("Failed to load model with path: %s", modelString.c_str());
    return -1;
  }

  // Load meta
  std::string name;
  double renderDistance;
  if (!private_nh.getParam("Name", name) ||
      !private_nh.getParam("RenderDistance", renderDistance)) {
    ROS_FATAL("Missing meta param");
    ROS_BREAK();
  }

  // Load transformations
  int initRotX, initRotY, initRotZ;
  double xTrans, yTrans, zTrans;
  double uniformScale;
  if (!private_nh.getParam("InitialRotX", initRotX) ||
      !private_nh.getParam("InitialRotY", initRotY) ||
      !private_nh.getParam("InitialRotZ", initRotZ) ||
      !private_nh.getParam("RelativeTranslateX", xTrans) ||
      !private_nh.getParam("RelativeTranslateY", yTrans) ||
      !private_nh.getParam("RelativeTranslateZ", zTrans) ||
      !private_nh.getParam("UniformScale", uniformScale)) {
    ROS_FATAL("Missing a transform param");
    ROS_BREAK();
  }

  // Tranform model...
  osg::ref_ptr<osg::PositionAttitudeTransform> transform =
      new osg::PositionAttitudeTransform;

  // Translate model
  osg::Vec3d translate(xTrans, yTrans, zTrans);
  transform->setPosition(translate);

  // Rotate the model
  rotate(transform, initRotX * degToRad, initRotY * degToRad,
         initRotZ * degToRad);  // Set initial pose

  // Set scale
  osg::Vec3d scale(uniformScale, uniformScale, uniformScale);
  transform->setScale(scale);

  // Attach transformation to camera scene
  if (!camera->addChild(transform.get())) {
    ROS_FATAL("Failed to add tranform to render to texture camera");
    return -1;
  }

  // Attach model to transformation
  if (!transform->addChild(model.get())) {
    ROS_FATAL("Failed to add model to render to texture camera");
    return -1;
  }

  // Attach camera to a root node
  // The previous camera was an indirection step
  // just for rendering to an image
  // if we want to render to the viewer window
  // we need to add children to the viewer's root camera
  osg::ref_ptr<osg::Group> root = new osg::Group;
  if (!root->addChild(camera.get())) {
    ROS_FATAL("Could not add render to texture camera as child of root node");
    return -1;
  }

  // Setup the viewer (attaching root to viewer)
  osgViewer::Viewer viewer;
  viewer.setSceneData(root.get());

  // Set viewer to run in a window so it can't brick our session
  viewer.setUpViewInWindow(0, 0, 300, 300);

  // Seems we need to render multiple times to ensure the texture is rendered...
  // At least to prevent a segfault
  viewer.frame();
  viewer.frame();

  // Check CUDA and OpenCV
  initCudaAndOpenCv();

  // Make sp filter
  SuperPixelFilter spFilter;
  spFilter.initialize(private_nh);

  // Make contour renderer
  ContourRenderer cRenderer;
  cRenderer.initialize(textureWidth, textureHeight);

  // Load filters
  std::vector<ColorRange> filters = getFiltersFromParams(private_nh);

  // Validate filters (they cannot fully contain each other)
  validateFilters(filters);

  // Load rotation ranges
  int xRotStart, xRotEnd, yRotStart, yRotEnd, zRotStart, zRotEnd, rotStep;
  if (!private_nh.getParam("StartRotateX", xRotStart) ||
      !private_nh.getParam("StartRotateY", yRotStart) ||
      !private_nh.getParam("StartRotateZ", zRotStart) ||
      !private_nh.getParam("EndRotateX", xRotEnd) ||
      !private_nh.getParam("EndRotateY", yRotEnd) ||
      !private_nh.getParam("EndRotateZ", zRotEnd) ||
      !private_nh.getParam("RotateStep", rotStep)) {
    ROS_FATAL("Missing a rotation range / step param");
    ROS_BREAK();
  }

  // Validate rotation ranges (start and ends cannot be the same)
  if (xRotStart == xRotEnd || yRotStart == yRotEnd || zRotStart == zRotEnd) {
    ROS_FATAL("Rotation start and end points must not be equal");
    return -1;
  }

  // Load frame rate and gSLICr usage
  bool gSLICrOn;
  int spinRate;
  int minimumContourArea;
  double contourLinearizationEpsilon;
  if (!private_nh.getParam("Fps", spinRate) ||
      !private_nh.getParam("gSLICrOn", gSLICrOn) ||
      !private_nh.getParam("ContourLinearizationEpsilon",
                           contourLinearizationEpsilon) ||
      !private_nh.getParam("MinimumContourArea", minimumContourArea)) {
    ROS_FATAL("Missing an optimization");
    ROS_BREAK();
  }
  ros::Rate rate(spinRate ? spinRate : 1);

  // Set up grids
  ShapeDb db;
  db.frameBufferWidth = textureWidth;
  db.frameBufferHeight = textureHeight;
  db.squareSize = gridSquareSize;
  db.gridRows = db.frameBufferHeight / gridSquareSize;
  db.gridCols = db.frameBufferWidth / gridSquareSize;
  db.renderDistance = renderDistance;

  ROS_ASSERT(textureWidth % gridSquareSize == 0);
  ROS_ASSERT(textureHeight % gridSquareSize == 0);
  // Main loop (process frames)
  int maxContoursInFrame = 0;
  int shapesProcessed = 0;
  for (int xRot = initRotX + xRotStart;
       xRot < xRotEnd + initRotX && !viewer.done() && !au_core::exitFlag;
       xRot += rotStep) {
    for (int yRot = initRotY + yRotStart;
         yRot < yRotEnd + initRotY && !viewer.done() && !au_core::exitFlag;
         yRot += rotStep) {
      for (int zRot = initRotZ + zRotStart;
           zRot < zRotEnd + initRotZ && !viewer.done() && !au_core::exitFlag;
           zRot += rotStep) {
        // Apply rotation
        rotate(transform, xRot * degToRad, yRot * degToRad, zRot * degToRad);

        // Render to texture
        viewer.frame();

        // Convert from OpenGL to OpenCV
        cv::Mat rendered(image->t(), image->s(), CV_8UC3);
        memcpy(rendered.data, image->data(), image->getTotalSizeInBytes());
        cv::Mat flippedTemp;
        cv::flip(rendered, flippedTemp, 0);
        rendered = flippedTemp;

        // Process the rendered image
        // NOTE: The background is augmented around the model, so a single color
        // background filter will not work. Use a wider one.
        cv::Mat viewOut, viewOutProcessed;
        int thisShapesAdded = calcShapesFromImage(
            rendered, filters, spFilter, cRenderer, xRot, yRot, zRot, name,
            viewOut, viewOutProcessed, gSLICrOn, db, minimumContourArea,
            contourLinearizationEpsilon);

        // Check that there are new compositions to register
        if (thisShapesAdded) {
          if (thisShapesAdded > maxContoursInFrame) {
            maxContoursInFrame = thisShapesAdded;
          }

          // Keep new shapes
          shapesProcessed += thisShapesAdded;
        }

        // Publish rendered image over ROS
        renderedPub.publish(
            cv_bridge::CvImage(std_msgs::Header(), "rgb8", rendered)
                .toImageMsg());

        // Tag output view with rotation status
        std::stringstream status;
        int lineHeight = 40;
        cv::Point writeHeader(5, lineHeight);
        status << "Rotation Status: (Step = " << rotStep
               << " & Target FPS = " << spinRate << ")";
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);
        status << "X [" << xRotStart << ", " << xRotEnd << "]: " << xRot;
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);
        status << "Y [" << yRotStart << ", " << yRotEnd << "]: " << yRot;
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);
        status << "Z [" << zRotStart << ", " << zRotEnd << "]: " << zRot;
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);
        status << shapesProcessed << " shapes processed";
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);
        status << "Max contours in a single frame: " << maxContoursInFrame;
        writeTextAndClearSs(status, viewOut, writeHeader, lineHeight);

        // Output view
        contourPub.publish(
            cv_bridge::CvImage(std_msgs::Header(), "rgb8", viewOut)
                .toImageMsg());
        contourProcessedPub.publish(
            cv_bridge::CvImage(std_msgs::Header(), "rgb8", viewOutProcessed)
                .toImageMsg());

        // Spin and idle
        ros::spinOnce();
        if (spinRate) {
          rate.sleep();
        }
      }
    }
  }

  // Write the database
  std::string path = ros::package::getPath("au_vision") + "/shape_dbs/newDB.sa";
  ROS_INFO("Writing to %s", path.c_str());
  saveShapeAnalysisDatabase(path, db);

  // Return
  return 0;
}

// Makes quaternions for each axis, multiplies them, then applies the
// transformation to the passed object
void rotate(osg::ref_ptr<osg::PositionAttitudeTransform>& transform,
            double xRot, double yRot, double zRot) {
  // Multiply quaternions to get transformation
  // Note that rotation is in radians
  osg::Quat xRotQuat, yRotQuat, zRotQuat;
  xRotQuat.makeRotate(xRot, osg::X_AXIS);
  yRotQuat.makeRotate(yRot, osg::Y_AXIS);
  zRotQuat.makeRotate(zRot, osg::Z_AXIS);
  osg::Quat xyzRotQuat = xRotQuat * yRotQuat * zRotQuat;

  // Apply quaternion
  transform->setAttitude(xyzRotQuat);
}

// Writes the text in a string stream to a cv::Mat and clears the string stream
// Also increments the passed point so we are ready for the next line
void writeTextAndClearSs(std::stringstream& text, cv::Mat& image,
                         cv::Point& where, int lineHeight) {
  cv::putText(image, text.str(), where, cv::FONT_HERSHEY_SIMPLEX, 1,
              cv::Scalar(255, 255, 255), 2);
  text.str(std::string());
  text.clear();
  where.y += lineHeight;
}

// Get filters from params file
std::vector<ColorRange> getFiltersFromParams(ros::NodeHandle& private_nh) {
  int filterCount;
  if (!private_nh.getParam("FilterCount", filterCount)) {
    ROS_FATAL("Missing filter count param");
    ROS_BREAK();
  }
  std::vector<ColorRange> filters(filterCount);
  ROS_INFO("Reading %i filters for image processing", filterCount);
  for (int i = 0; i < filterCount; ++i) {
    std::string mode;  // Include / Exclude
    if (!private_nh.getParam(
            std::string("Filter_") + std::to_string(i) + "_Mode", mode)) {
      ROS_FATAL("Filter is missing mode");
      ROS_BREAK();
    }
    if (mode == "include" || mode == "exclude") {
      cv::Scalar target;
      int margin;
      if (!private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_L", target[0]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_A", target[1]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_B", target[2]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_Margin", margin)) {
        ROS_FATAL("Missing a filter param");
        ROS_BREAK();
      }
      if (mode == "include") {
        filters[i] = ColorRange(target, margin);
      } else if (mode == "exclude") {
        ColorRange targetRange(target, margin);
        ColorRange secondHalf;
        for (int w = 0; w < 3; ++w) {
          filters[i].lower[w] = 0;
          filters[i].upper[w] = targetRange.lower[w];
          secondHalf.upper[w] = 255;
          secondHalf.lower[w] = targetRange.upper[w];
        }
        filters.insert(filters.begin() + (++i), secondHalf);
      }
    } else if (mode == "range") {
      if (!private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_LowerL",
              filters[i].lower[0]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_LowerA",
              filters[i].lower[1]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_LowerB",
              filters[i].lower[2]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_UpperL",
              filters[i].upper[0]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_UpperA",
              filters[i].upper[1]) ||
          !private_nh.getParam(
              std::string("Filter_") + std::to_string(i) + "_UpperB",
              filters[i].upper[2])) {
        ROS_FATAL("Missing a filter param");
        ROS_BREAK();
      }
    } else {
      ROS_FATAL("Filter has invalid mode");
      ROS_BREAK();
    }
  }

  // Return
  return filters;
}

// Checks to make sure that no filters fully contain each other
void validateFilters(std::vector<ColorRange>& filters) {
  for (int i = 0; i < filters.size(); ++i) {
    for (int j = 0; j < filters.size(); ++j) {
      int nested = 0;
      for (int k = 0; k < 3; ++k) {
        if (filters[i].lower[k] > filters[j].lower[k] &&
            filters[i].upper[k] < filters[j].upper[k]) {
          ++nested;
        }
      }
      if (nested == 3) {
        ROS_FATAL("Filters %i is nested in filter %i. This is not allowed.", i,
                  j);
        ROS_BREAK();
      }
    }
  }
}

int calcShapesFromImage(const cv::Mat& imageRgb,
                        const std::vector<ColorRange>& filters,
                        SuperPixelFilter& spFilter, ContourRenderer& renderer,
                        int rotX, int rotY, int rotZ, const std::string& name,
                        cv::Mat& viewOut, cv::Mat& viewOutProcessed,
                        bool gSLICrOn, ShapeDb& db, int minimumContourArea,
                        double contourLinearizationEpsilon) {
  ROS_ASSERT(!imageRgb.empty());
  ROS_ASSERT(imageRgb.channels() == 3);

  // Count new shapes
  int shapesAdded = 0;

  // Convert to LAB
  cv::cuda::GpuMat dev_image, dev_imageRgb(imageRgb);
  cv::Mat image;
  cv::cuda::cvtColor(dev_imageRgb, dev_image, cv::COLOR_RGB2Lab);
  dev_image.download(image);

  // Run gSLICr
  cv::Mat colorMask;
  if (gSLICrOn) {
    spFilter.filterLabImage(image);

    // Get solid color spixels
    std::vector<cv::Scalar> colorList;
    spFilter.resultAverageColorMask(colorMask, colorList);
  } else {
    colorMask = image;
  }

  // Output image
  cv::Mat totalOut(colorMask.rows, colorMask.cols, CV_8UC3, cv::Scalar(0));

  // Cycle over filters
  std::vector<Shape> shapes;
  cv::Rect groupBound;
  for (auto filter : filters) {
    // Get mask
    cv::cuda::GpuMat dev_colorMask(colorMask), dev_rangeMask;
    callInRange_device(dev_colorMask, filter.lower, filter.upper,
                       dev_rangeMask);
    cudaDeviceSynchronize();
    cv::Mat rangeMask;
    dev_rangeMask.download(rangeMask);

    // Get contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(rangeMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // Cull contours
    cullContours(contours, rangeMask.cols, rangeMask.rows, minimumContourArea);

    // Simplify contours
    for (int i = 0; i < contours.size(); ++i) {
      // Higher epsilon is more linear / simple
      cv::approxPolyDP(contours[i], contours[i], contourLinearizationEpsilon,
                       true);
    }

    // TODO: Remove duplicate contours (filters might have unions)

    // Scale and center contours for drawing
    std::vector<std::vector<cv::Point>> processedContours;
    for (auto c : contours) {
      std::vector<cv::Point> newC = c;
      centerAndMaximizeContour(newC, renderer.width(), renderer.height());
      processedContours.emplace_back(newC);
    }

    // Draw contours
    cv::cuda::cvtColor(dev_rangeMask, dev_rangeMask, cv::COLOR_GRAY2RGB);
    dev_rangeMask.download(viewOut);
    viewOutProcessed =
        cv::Mat(viewOut.rows, viewOut.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::drawContours(viewOut, contours, -1, filter.lower, 4);
    cv::drawContours(viewOutProcessed, processedContours, -1,
                     cv::Scalar(255, 255, 255), CV_FILLED);

    // Draw points on contours
    for (int i = 0; i < contours.size(); ++i) {
      for (int j = 0; j < contours[i].size(); ++j) {
        cv::circle(viewOut, contours[i][j], 3, filter.upper, 3);
      }
    }

    totalOut += viewOut;

    // Get shapes
    for (int c = 0; c < processedContours.size(); ++c) {
      // Get contour color (at first point, because hack)
      // TODO: Get contour average color
      cv::Scalar thisColor;
      for (int comp = 0; comp < 3; ++comp) {
        thisColor[comp] =
            colorMask.at<cv::Vec3b>(contours[c][0].y, contours[c][0].x)[comp];
      }

      Shape newShape;
      newShape.name = "Part";
      newShape.color = thisColor;
      newShape.contour = std::move(processedContours[c]);
      newShape.area = cv::contourArea(contours[c]);

      // Get contour bound
      cv::Rect thisBound = cv::boundingRect(contours[c]);
      newShape.bound.x = thisBound.x;
      newShape.bound.y = thisBound.y;
      newShape.bound.width = thisBound.width;
      newShape.bound.height = thisBound.height;

      // Add to group bound
      if (shapesAdded == 0) {
        groupBound = thisBound;
      } else {
        groupBound |= thisBound;
      }

      ++shapesAdded;
      shapes.emplace_back(newShape);
    }
  }

  // Make pose
  cv::Scalar pose;
  pose[0] = rotX;
  pose[1] = rotY;
  pose[2] = rotZ;

  // Convert shape offsets to final group bound
  for (auto& s : shapes) {
    int pre = s.bound.x;
    s.bound.x -= groupBound.x;
    s.bound.y -= groupBound.y;
  }

  // Add group to DB
  if (shapes.size()) {
    db.addGroup(shapes, renderer, name, pose, groupBound);
  }

  // Return
  viewOut = totalOut;
  return shapesAdded;
}
