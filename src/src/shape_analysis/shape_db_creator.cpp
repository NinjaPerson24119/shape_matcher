#include <au_core/sigint_handler.h>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glcorearb.h>

#include <au_vision/shape_analysis/gpu_util.h>
#include <au_vision/shape_analysis/shape_analysis.h>
#include <au_vision/shape_analysis/superpixel_filter.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/opencv.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <osg/Camera>
#include <osg/Image>
#include <osg/PositionAttitudeTransform>
#include <osg/Quat>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>

// recommended OSG book: OpenSceneGraph 3.0 Beginner's Guide  (online docs
// aren't that great)

using namespace au_vision;

const double degToRad = osg::PI / 180;

// Predeclarations
void rotate(osg::ref_ptr<osg::PositionAttitudeTransform>& transform,
            cv::Scalar pose);

void writeTextAndClearSs(std::stringstream& text, cv::Mat& image,
                         cv::Point& where, int lineHeight);

std::vector<ColorRange> getFiltersFromParams(ros::NodeHandle& private_nh);

void validateFilters(std::vector<ColorRange>& vfilters);

int calcShapesFromImage(const cv::Mat& image, SuperPixelFilter& spFilter,
                        cv::Scalar poseIn, cv::Mat& viewOut,
                        cv::Mat& viewOutProcessed);

void workerFunc(int id);

// Globals
ShapeDb db;
std::string name;
double renderDistance;
int morphologyFilterSize;
bool gSLICrOn;
double contourLinearizationEpsilon;
int minimumContourArea;
std::vector<ColorRange> filters;
ros::NodeHandle* private_nh;
std::string nsPrefix;
std::atomic<int> maxContoursInFrame;
std::atomic<int> shapesProcessed;
int threadSleepMs = 10;
int textureWidth, textureHeight;
bool debug;

// Worker variables
std::mutex dbMutex;
std::vector<cv::Mat> w_image;
std::vector<cv::Mat> w_rgbImage;
std::vector<cv::Scalar> w_poseIn;
std::vector<cv::Mat> w_viewOut;
std::vector<cv::Mat> w_viewOutProcessed;

// -1 = kill, 0 = idle, 1 = running, 2 = done
std::atomic<int>* w_flags;
std::vector<std::thread> workers;

// Entry point
int main(int argc, char** argv) {
  // Private node handle
  au_core::handleInterrupts(argc, argv, "shape_db_creator", true);
  private_nh = new ros::NodeHandle("~");

  // Set up ROS out
  image_transport::ImageTransport imageTransport(*private_nh);
  image_transport::Publisher renderedPub =
      imageTransport.advertise("transformView", 10);
  image_transport::Publisher contourPub =
      imageTransport.advertise("contourView", 10);
  image_transport::Publisher contourProcessedPub =
      imageTransport.advertise("contourProcessedView", 10);

  // param namespace prefix
  std::string dbName;
  if (!private_nh->getParam("db", dbName)) {
    ROS_FATAL("DB name was not passed");
    return -1;
  }
  nsPrefix = dbName + "_db_creator/";

  // Load FBO info (this is read first since OpenSceneGraph depends on
  // it)
  int gridSquareSize;
  if (!private_nh->getParam(nsPrefix + "gSLICr_width", textureWidth) ||
      !private_nh->getParam(nsPrefix + "gSLICr_height", textureHeight) ||
      !private_nh->getParam(nsPrefix + "SquareSize", gridSquareSize)) {
    ROS_FATAL("FBO param: missing gSLICr dims / FBO dims");
    ROS_BREAK();
  }

  // Load clear color
  cv::Scalar background;
  if (!private_nh->getParam(nsPrefix + "BackgroundR", background[0]) ||
      !private_nh->getParam(nsPrefix + "BackgroundG", background[1]) ||
      !private_nh->getParam(nsPrefix + "BackgroundB", background[2])) {
    ROS_FATAL("Missing background color");
    ROS_BREAK();
  }
  for (int i = 0; i < 3; ++i) {
    background[i] /= 255;
  }

  // Load debug state
  if (!private_nh->getParam("debug", debug)) {
    ROS_FATAL("Debug arg was not passed");
    return -1;
  }

  // Load model
  std::string modelString;
  if (!private_nh->getParam("model", modelString)) {
    ROS_FATAL("Model path was not passed");
    return -1;
  }
  osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(modelString);
  if (!model.get()) {
    ROS_FATAL("Failed to load model with path: %s", modelString.c_str());
    return -1;
  }

  // Load meta
  if (!private_nh->getParam(nsPrefix + "Name", name) ||
      !private_nh->getParam(nsPrefix + "RenderDistance", renderDistance) ||
      !private_nh->getParam(nsPrefix + "MorphologyFilterSize",
                            morphologyFilterSize)) {
    ROS_FATAL("Missing meta param");
    ROS_BREAK();
  }
  if (morphologyFilterSize % 2 == 0) {
    ROS_FATAL("Morphology filter size must be odd");
    ROS_BREAK();
  }

  // Load transformations
  cv::Scalar initRot, initTrans;
  double uniformScale;
  if (!private_nh->getParam(nsPrefix + "InitialRotX", initRot[0]) ||
      !private_nh->getParam(nsPrefix + "InitialRotY", initRot[1]) ||
      !private_nh->getParam(nsPrefix + "InitialRotZ", initRot[2]) ||
      !private_nh->getParam(nsPrefix + "RelativeTranslateX", initTrans[0]) ||
      !private_nh->getParam(nsPrefix + "RelativeTranslateY", initTrans[1]) ||
      !private_nh->getParam(nsPrefix + "RelativeTranslateZ", initTrans[2]) ||
      !private_nh->getParam(nsPrefix + "UniformScale", uniformScale)) {
    ROS_FATAL("Missing a transform param");
    ROS_BREAK();
  }

  // Load filters
  filters = getFiltersFromParams(*private_nh);

  // Validate filters (they cannot fully contain each other)
  validateFilters(filters);

  // Load rotation ranges
  cv::Scalar rotStart, rotEnd;
  int rotStep;
  if (!private_nh->getParam(nsPrefix + "StartRotateX", rotStart[0]) ||
      !private_nh->getParam(nsPrefix + "StartRotateY", rotStart[1]) ||
      !private_nh->getParam(nsPrefix + "StartRotateZ", rotStart[2]) ||
      !private_nh->getParam(nsPrefix + "EndRotateX", rotEnd[0]) ||
      !private_nh->getParam(nsPrefix + "EndRotateY", rotEnd[1]) ||
      !private_nh->getParam(nsPrefix + "EndRotateZ", rotEnd[2]) ||
      !private_nh->getParam(nsPrefix + "RotateStep", rotStep)) {
    ROS_FATAL("Missing a rotation range / step param");
    ROS_BREAK();
  }

  // Validate rotation ranges (start and ends cannot be the same)
  for (int i = 0; i < 3; ++i) {
    if (rotStart[i] == rotEnd[i]) {
      ROS_FATAL("Rotation start and end points must not be equal");
      return -1;
    }
  }

  // Load frame rate and gSLICr usage
  int spinRate;
  if (!private_nh->getParam(nsPrefix + "Fps", spinRate) ||
      !private_nh->getParam(nsPrefix + "gSLICrOn", gSLICrOn) ||
      !private_nh->getParam(nsPrefix + "ContourLinearizationEpsilon",
                            contourLinearizationEpsilon) ||
      !private_nh->getParam(nsPrefix + "MinimumContourArea",
                            minimumContourArea)) {
    ROS_FATAL("Missing an optimization");
    ROS_BREAK();
  }
  ros::Rate rate(spinRate ? spinRate : 1);

  // Set up grids
  db.frameBufferWidth = textureWidth;
  db.frameBufferHeight = textureHeight;
  db.squareSize = gridSquareSize;
  db.gridRows = db.frameBufferHeight / gridSquareSize;
  db.gridCols = db.frameBufferWidth / gridSquareSize;
  db.renderDistance = renderDistance;

  // Verify grid size is set properly for rough area differences
  ROS_ASSERT(textureWidth % gridSquareSize == 0);
  ROS_ASSERT(textureHeight % gridSquareSize == 0);

  // calculate how many renders we can do per batch
  // divide by 2 because we need frame memory on both device and host (TX2
  // shares memory) usage breakdown per OSG camera:
  // - 24-bits for color buffer (GL_RGB on GPU)
  // - 24-bits for depth buffer
  // - 24-bits for color buffer (CPU RGB) (CPU CIELAB) (OpenCV GPU)
  // hence, multiply by 15 (total channels of uchar)
  size_t RGB_maxMemory = 2.0e9;
  int RGB_parallelFrames =
      (RGB_maxMemory / 2) / (textureWidth * textureHeight * 15);
  ROS_ASSERT(RGB_parallelFrames);

  // Do anti aliasing
  osg::DisplaySettings::instance()->setNumMultiSamples(4);

  // Setup images, cameras, viewers, transforms
  ROS_INFO(
      "RGB Renderer limited to roughly %0.6f MB. Rendering color images in "
      "batches of %i",
      (double)RGB_maxMemory / 1.0e6, RGB_parallelFrames);
  std::vector<osg::ref_ptr<osg::Texture2D>> images;
  std::vector<osg::ref_ptr<osg::Camera>> cameras;
  std::vector<osg::ref_ptr<osg::PositionAttitudeTransform>> transforms;
  osg::ref_ptr<osg::Group> root = new osg::Group();
  osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer();

  // set threading mode to single (so that frame() blocks)
  viewer->setThreadingModel(osgViewer::ViewerBase::SingleThreaded);

  // set viewer to run in a window so it can't brick our session
  // (default is full screen)
  viewer->setUpViewInWindow(0, 0, 50, 50);

  // realize
  viewer->realize();

  // Check CUDA and OpenCV
  initCudaAndOpenCv();

  // get context (making a custom one is a joke and doesn't work at all in osg's
  // framework)
  osg::GraphicsContext* context;
  std::vector<osg::GraphicsContext*> ctxs;
  viewer->getContexts(ctxs, true);
  ROS_ASSERT(ctxs.size());
  context = ctxs[0];
  ROS_ASSERT(context->makeCurrent());

  for (int i = 0; i < RGB_parallelFrames; ++i) {
    // allocate image
    images.emplace_back(osg::ref_ptr<osg::Texture2D>(new osg::Texture2D()));
    images[i]->setTextureSize(textureWidth, textureHeight);
    images[i]->setInternalFormat(GL_RGB);
    images[i]->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
    images[i]->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);

    // make camera
    cameras.emplace_back(osg::ref_ptr<osg::Camera>(new osg::Camera()));

    // attach context
    cameras[i]->setGraphicsContext(context);

    // render with frame buffer
    cameras[i]->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);

    // bind image to the camera's color buffer (render to image)
    cameras[i]->attach(osg::Camera::COLOR_BUFFER, images[i].get());

    // set absolute reference frame
    cameras[i]->setReferenceFrame(osg::Camera::ABSOLUTE_RF);

    // render (to image) prior to sending to viewer
    cameras[i]->setRenderOrder(osg::Camera::PRE_RENDER);

    // set viewport to texture size
    cameras[i]->setViewport(0, 0, textureWidth, textureHeight);

    // set clear color and buffers to be cleared
    cameras[i]->setClearColor(
        osg::Vec4(background[0], background[1], background[2], 0.0f));
    cameras[i]->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // allocate transform
    transforms.emplace_back(osg::ref_ptr<osg::PositionAttitudeTransform>(
        new osg::PositionAttitudeTransform));

    // set translation and scale. we only need to do this once since only
    // rotation will vary
    osg::Vec3d translate(initTrans[0], initTrans[1], initTrans[2]);
    transforms[i]->setPosition(translate);
    osg::Vec3d scale(uniformScale, uniformScale, uniformScale);
    transforms[i]->setScale(scale);

    // attach model to transformation
    if (!transforms[i]->addChild(model.get())) {
      ROS_FATAL("Failed to add the model to a transform");
      return -1;
    }

    // attach transformation (which is attached to the model) to the camera
    if (!cameras[i]->addChild(transforms[i].get())) {
      ROS_FATAL("Failed to add a tranform to render to texture camera");
      return -1;
    }

    // attach cameras to root
    root->addChild(cameras[i].get());
  }

  // attach camera
  viewer->setSceneData(root.get());

  // actually build textures on first pass
  viewer->frame();

  // get number of existing hardware threads
  // use n-1 of them to retain one thread for basic responsiveness
  unsigned int coresMinus1 = (std::thread::hardware_concurrency() - 1);
  unsigned int hardwareThreads = coresMinus1 > 0 ? coresMinus1 : 1;
  ROS_INFO("Processing renders on CPU in batches of %i (cores)",
           hardwareThreads);

  // resize parameter vectors
  w_image.resize(hardwareThreads);
  w_rgbImage.resize(hardwareThreads);
  w_poseIn.resize(hardwareThreads);
  w_viewOut.resize(hardwareThreads);
  w_viewOutProcessed.resize(hardwareThreads);
  w_flags = new std::atomic<int>[hardwareThreads];
  for (int j = 0; j < hardwareThreads; ++j) {
    // idle all threads
    w_flags[j] = 0;
  }

  // run threads
  for (int i = 0; i < hardwareThreads; ++i) {
    workers.emplace_back(std::thread(workerFunc, i));
  }

  // read back prep (CUDA)
  std::vector<cv::Mat> cMats, coMats;
  std::vector<cv::cuda::GpuMat> gMats;
  for (int i = 0; i < RGB_parallelFrames; ++i) {
    cMats.push_back(
        cv::Mat(textureHeight, textureWidth, CV_8UC3, cv::Scalar(0, 255, 0)));
    coMats.push_back(
        cv::Mat(textureHeight, textureWidth, CV_8UC3, cv::Scalar(0, 255, 0)));
    gMats.push_back(cv::cuda::GpuMat(textureHeight, textureWidth, CV_8UC3));
  }

  // generate PBOs
  ROS_ASSERT(context->makeCurrent());
  size_t imageBytes = textureWidth * textureHeight * 3;
  std::vector<GLuint> pbo(RGB_parallelFrames);
  glGenBuffers(RGB_parallelFrames, &pbo[0]);
  for (int i = 0; i < RGB_parallelFrames; ++i) {
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
    glBufferData(GL_PIXEL_PACK_BUFFER, imageBytes, nullptr, GL_DYNAMIC_READ);
  }

  // set pixel storage info for readback
  // without setting the alignment to 1, OGL will pack RGB pixels into 4 bytes
  // but our buffers are only sized for 3 channels (3 bytes per pixel)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ROW_LENGTH, textureWidth);

  // main loop
  cv::Scalar iterRot = initRot + rotStart;
  std::vector<cv::Scalar> poses;
  maxContoursInFrame = 0;
  shapesProcessed = 0;
  bool loop = true;
  while (loop && !au_core::exitFlag) {
    // configure batch
    poses.clear();
    for (int k = 0; k < RGB_parallelFrames; ++k) {
      // apply rotation
      rotate(transforms[k], iterRot);
      poses.push_back(iterRot);

      // increment rotations
      int r = 2;
      while (true) {
        iterRot[r] += rotStep;
        // check if axis has spun fully
        if (iterRot[r] >= rotEnd[r] + initRot[r]) {
          // reset this axis
          iterRot[r] = initRot[r] + rotStart[r];

          // spin next axis
          --r;

          // check for exit condition
          if (r == -1) {
            loop = false;
            break;
          }
        } else {
          // otherwise keep spinning
          break;
        }
      }
    }

    // render batch to textures
    // note: if vsync is on, the speed of this will be limited
    viewer->frame();

    // batch download textures to GPU global mem
    int ctx = context->getState()->getContextID();
    for (int c = 0; c < poses.size(); ++c) {
      // copy OpenGL to OpenCV
      osg::Texture::TextureObject* texObj = images[c]->getTextureObject(ctx);
      if (texObj == nullptr) {
        ROS_FATAL("Bad texture ID");
        ROS_BREAK();
      }
      ROS_ASSERT(context->makeCurrent());

      // bind PBO and texture
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[c]);
      glBindTexture(GL_TEXTURE_2D, texObj->_id);

      // read texture into PBO
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }

    // map PBO for reading, copy to OpenCV, unmap PBO
    // note: this will block until the texture readback is complete
    for (int c = 0; c < poses.size(); ++c) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[c]);
      unsigned char* src = static_cast<unsigned char*>(
          glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));
      ROS_ASSERT(src != nullptr);
      memcpy(coMats[c].data, src, imageBytes);
      glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    // bulk operate matrices
    cv::cuda::Stream cvStream;

    // upload to GPU
    for (int c = 0; c < poses.size(); ++c) {
      gMats[c].upload(coMats[c], cvStream);
    }
    cvStream.waitForCompletion();

    // convert to CIELAB
    for (int c = 0; c < poses.size(); ++c) {
      cv::cuda::cvtColor(gMats[c], gMats[c], cv::COLOR_RGB2Lab, 0, cvStream);
    }
    cvStream.waitForCompletion();

    // download back to CPU
    for (int c = 0; c < poses.size(); ++c) {
      gMats[c].download(cMats[c], cvStream);
    }
    cvStream.waitForCompletion();

    // process rendered textures
    for (int t = 0; t < poses.size() && !au_core::exitFlag;
         t += hardwareThreads) {
      // trigger threads
      int running = 0;
      for (int k = t; k < t + hardwareThreads && k < poses.size(); ++k) {
        int batch = k - t;
        int f = w_flags[batch].load(std::memory_order_relaxed);
        assert(f == 0);

        // Convert from OpenGL to OpenCV (flip image)
        cv::flip(cMats[k], w_image[batch], 0);
        cv::flip(coMats[k], w_rgbImage[batch], 0);

        // load params
        w_poseIn[batch] = poses[k] - initRot;

        // unleash
        w_flags[batch].store(1, std::memory_order_relaxed);
        ++running;
      }

      // wait for completion
      while (true) {
        int numDone = 0;
        for (int j = 0; j < hardwareThreads; ++j) {
          int f = w_flags[j].load(std::memory_order_relaxed);
          if (f == 2) {
            ++numDone;
          }
        }
        if (numDone == running) {
          break;
        }

        // avoid using 100% CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(threadSleepMs));
      }

      // handle outputs
      for (int j = 0; j < hardwareThreads; ++j) {
        // Publish rendered image over ROS
        renderedPub.publish(
            cv_bridge::CvImage(std_msgs::Header(), "rgb8", w_rgbImage[j])
                .toImageMsg());

        // Tag output view with rotation status
        std::stringstream status;
        int lineHeight = 40;
        cv::Point writeHeader(5, lineHeight);
        status << "Rotation Status: (Step = " << rotStep
               << " & Target FPS = " << spinRate << ")";
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);
        status << "X [" << rotStart[0] << ", " << rotEnd[0]
               << "]: " << w_poseIn[j][0];
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);
        status << "Y [" << rotStart[1] << ", " << rotEnd[1]
               << "]: " << w_poseIn[j][1];
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);
        status << "Z [" << rotStart[2] << ", " << rotEnd[2]
               << "]: " << w_poseIn[j][2];
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);
        status << shapesProcessed << " shapes processed";
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);
        status << "Max contours in a single frame: " << maxContoursInFrame;
        writeTextAndClearSs(status, w_viewOut[j], writeHeader, lineHeight);

        // Output view
        contourPub.publish(
            cv_bridge::CvImage(std_msgs::Header(), "rgb8", w_viewOut[j])
                .toImageMsg());
        contourProcessedPub.publish(cv_bridge::CvImage(std_msgs::Header(),
                                                       "rgb8",
                                                       w_viewOutProcessed[j])
                                        .toImageMsg());

        // Spin and idle
        // note, even though the batch is done, this will enforce the FPS
        ros::spinOnce();
        if (spinRate) {
          rate.sleep();
        }

        // set thread to idle
        w_flags[j].store(0, std::memory_order_relaxed);
      }
    }
  }

  // kill threads
  for (int i = 0; i < workers.size(); ++i) {
    w_flags[i].store(-1, std::memory_order_relaxed);
    workers[i].join();
  }
  delete[] w_flags;

  // Write the database
  std::string path = ros::package::getPath("au_vision") + "/shape_dbs/newDB.sa";
  ROS_INFO("Writing to %s", path.c_str());
  saveShapeAnalysisDatabase(path, db);

  // deallocate
  glDeleteBuffers(RGB_parallelFrames, &pbo[0]);
  delete private_nh;

  // Return
  return 0;
}

// Makes quaternions for each axis, multiplies them, then applies the
// transformation to the passed object
void rotate(osg::ref_ptr<osg::PositionAttitudeTransform>& transform,
            cv::Scalar pose) {
  // Multiply quaternions to get transformation
  // Note that rotation is in radians
  osg::Quat xRotQuat, yRotQuat, zRotQuat;
  xRotQuat.makeRotate(pose[0] * degToRad, osg::X_AXIS);
  yRotQuat.makeRotate(pose[1] * degToRad, osg::Y_AXIS);
  zRotQuat.makeRotate(pose[2] * degToRad, osg::Z_AXIS);
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
  // param namespace prefix
  std::string dbName;
  if (!private_nh.getParam("db", dbName)) {
    ROS_FATAL("DB name was not passed");
    ROS_BREAK();
  }
  std::string nsPrefix = dbName + "_db_creator/";

  int filterCount;
  if (!private_nh.getParam(nsPrefix + "FilterCount", filterCount)) {
    ROS_FATAL("Missing filter count param");
    ROS_BREAK();
  }
  std::vector<ColorRange> filtersv(filterCount);
  ROS_INFO("Reading %i filters for image processing", filterCount);
  for (int i = 0; i < filterCount; ++i) {
    std::string mode;  // Include / Exclude
    if (!private_nh.getParam(nsPrefix + "Filter_" + std::to_string(i) + "_Mode",
                             mode)) {
      ROS_FATAL("Filter is missing mode");
      ROS_BREAK();
    }
    if (mode == "include" || mode == "exclude") {
      cv::Scalar target;
      int margin;
      if (!private_nh.getParam(nsPrefix + "Filter_" + std::to_string(i) + "_L",
                               target[0]) ||
          !private_nh.getParam(nsPrefix + "Filter_" + std::to_string(i) + "_A",
                               target[1]) ||
          !private_nh.getParam(nsPrefix + "Filter_" + std::to_string(i) + "_B",
                               target[2]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_Margin", margin)) {
        ROS_FATAL("Missing a filter param");
        ROS_BREAK();
      }
      if (mode == "include") {
        filtersv[i] = ColorRange(target, margin);
      } else if (mode == "exclude") {
        ColorRange targetRange(target, margin);
        ColorRange secondHalf;
        for (int w = 0; w < 3; ++w) {
          filtersv[i].lower[w] = 0;
          filtersv[i].upper[w] = targetRange.lower[w];
          secondHalf.upper[w] = 255;
          secondHalf.lower[w] = targetRange.upper[w];
        }
        filtersv.insert(filtersv.begin() + (++i), secondHalf);
      }
    } else if (mode == "range") {
      if (!private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_LowerL",
              filtersv[i].lower[0]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_LowerA",
              filtersv[i].lower[1]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_LowerB",
              filtersv[i].lower[2]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_UpperL",
              filtersv[i].upper[0]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_UpperA",
              filtersv[i].upper[1]) ||
          !private_nh.getParam(
              nsPrefix + "Filter_" + std::to_string(i) + "_UpperB",
              filtersv[i].upper[2])) {
        ROS_FATAL("Missing a filter param");
        ROS_BREAK();
      }
    } else {
      ROS_FATAL("Filter has invalid mode");
      ROS_BREAK();
    }
  }

  // Return
  return filtersv;
}

// Checks to make sure that no filters fully contain each other
void validateFilters(std::vector<ColorRange>& vfilters) {
  for (int i = 0; i < vfilters.size(); ++i) {
    for (int j = 0; j < vfilters.size(); ++j) {
      if (i == j) continue;
      if (vfilters[i].contains(vfilters[j])) {
        ROS_FATAL("Filters %i is nested in filter %i. This is not allowed.", i,
                  j);
        ROS_BREAK();
      }
    }
  }
}

int calcShapesFromImage(const cv::Mat& image, SuperPixelFilter& spFilter,
                        cv::Scalar poseIn, cv::Mat& viewOut,
                        cv::Mat& viewOutProcessed) {
  ROS_ASSERT(!image.empty());
  ROS_ASSERT(image.channels() == 3);
  // it is assumed that 'image' in the CIELAB color space

  // Count new shapes
  int shapesAdded = 0;

  // Run gSLICr
  cv::Mat colorMask;
  if (gSLICrOn) {
    spFilter.filterLabImage(image);

    // Get solid color spixels
    std::vector<cv::Scalar> colorList;
    spFilter.resultColors(colorList);
    spFilter.resultAverageColorMask(colorMask, colorList);
  } else {
    colorMask = image;
  }

  // Output image
  cv::Mat totalOut(colorMask.rows, colorMask.cols, CV_8UC3, cv::Scalar(0));

  // Cycle over filters
  std::vector<Shape> shapes;
  cv::Rect groupBound;
  bool okGroup = true;
  for (auto filter : filters) {
    if (!okGroup) break;
    // Get mask
    cv::cuda::GpuMat dev_colorMask(colorMask), dev_rangeMask;
    dev_rangeMask.create(dev_colorMask.rows, dev_colorMask.cols, CV_8UC1);
    callInRange_device(dev_colorMask, filter.lower, filter.upper, dev_rangeMask,
                       0);
    cudaDeviceSynchronize();
    gpuErrorCheck(cudaGetLastError());  // Verify that all went OK
    cv::Mat rangeMask;
    dev_rangeMask.download(rangeMask);

    // Get contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(rangeMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // Cull contours
    // note that maxPoints=0 means as no cap
    cullContours(contours, rangeMask.cols, rangeMask.rows, minimumContourArea,
                 0, 0);

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
      centerAndMaximizeContour(newC, db.frameBufferWidth, db.frameBufferHeight);
      processedContours.emplace_back(newC);
    }

    // Draw contours
    if (debug) {
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
    }

    // Get shapes

    // Get interor and exterior contour colors
    std::vector<ShapeColor> colors;
    averageContourColors(colorMask, contours, colors, morphologyFilterSize);

    for (int c = 0; c < processedContours.size(); ++c) {
      // Only use colors that are inside the filter
      colors[c].validInterior =
          filter.contains(ColorRange(colors[c].interior, 0));
      colors[c].validExterior =
          filter.contains(ColorRange(colors[c].exterior, 0));
      if (!(colors[c].validInterior || colors[c].validExterior)) {
        std::cout << "COLOR "
                     "interior_color,exterior_color,useInteriorColor?,"
                     "useExteriorColor?: "
                  << colors[c].interior << ", " << colors[c].exterior << ", "
                  << colors[c].validInterior << ", " << colors[c].validExterior
                  << "\n";
        std::cout << "Skipping group\n";
        okGroup = false;
        break;
      }

      // DEBUG (but keep this for tuning later)
      // std::cout << "COLOR i,e,valid_i,valid_e: " << colors[c].interior << ",
      // " <<  colors[c].exterior << ", " << colors[c].validInterior << ", " <<
      // colors[c].validExterior << "\n";

      Shape newShape;
      newShape.name = "Part";
      newShape.color = colors[c];
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
  cv::Scalar pose = poseIn;

  // Convert shape offsets to final group bound
  for (auto& s : shapes) {
    int pre = s.bound.x;
    s.bound.x -= groupBound.x;
    s.bound.y -= groupBound.y;
  }

  // Add group to DB
  if (shapes.size() && okGroup) {
    dbMutex.lock();
    db.addGroup(shapes, name, pose, groupBound);
    dbMutex.unlock();
  }

  // Return
  viewOut = totalOut;
  return shapesAdded;
}

void workerFunc(int id) {
  // initialize gSLICr
  // memory for CUDA is per thread, so initialization must
  // occur in the applicable thread
  SuperPixelFilter spFilter;
  spFilter.initialize(*private_nh, nsPrefix);

  while (true) {
    // cache flag
    int flagCopy = w_flags[id].load(std::memory_order_relaxed);
    if (flagCopy == -1) {
      return;
    }
    if (flagCopy == 1) {
      // execute. assume params are loaded.

      // Process the rendered image
      // NOTE: The background is augmented around the model by artifacts,
      // aliasing, etc., so a single color background filter will not work.
      // Use a wider one.
      int thisShapesAdded =
          calcShapesFromImage(w_image[id], spFilter, w_poseIn[id],
                              w_viewOut[id], w_viewOutProcessed[id]);

      // Check that there are new compositions to register
      if (thisShapesAdded) {
        if (thisShapesAdded > maxContoursInFrame) {
          maxContoursInFrame = thisShapesAdded;
        }

        // Keep new shapes
        shapesProcessed += thisShapesAdded;
      }

      // flag done
      w_flags[id].store(2, std::memory_order_relaxed);
    }
  }
}