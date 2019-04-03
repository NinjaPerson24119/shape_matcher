# NOTES

Build with catkin_make in the root dir

This repo doesn't implement a method for passing images to the shape matcher, nor for rendering
its output.  It's only the algorithm. Use the functions in shape_analysis_detector_common.h

# Vision

This repo contains some of ARVP's image processing algorithms and vision tools. This includes:

* Computer Vision
  * [OpenCV / CUDA](#opencv--cuda)
  * [Superpixel Image Segmentation (gSLICr)](#superpixel-image-segmentation-gslicr)
  * [Automatic Filter](#automatic-filter)
  * [Shape Matcher](#shape-matcher)
  * [Shape Database Creator](#shape-db-creator)

## OpenCV / CUDA

The vision library is configured to download OpenCV 3.3.1 and compile it for CUDA support. Though the tracker GUI will list detectors, if the vision library is not compiled with CUDA support, detectors requiring CUDA will not be built if CUDA is not detected. The shape DB creator is also unavailable without CUDA. OpenCV is built to the "catkin_ws/build_external" folder so that it is not easily deleted (since its kernels take a while to compile).

Note that though the minimum accepted CUDA version is 8.0, the shape matcher code was developed on the latest patch of CUDA 9.0.176.

## [Superpixel Image Segmentation](https://docs.google.com/document/d/1S_obWjfPqkI1xsj9CWjRCkdUNJaTfv0qC-ndoh6PBR0/edit#heading=h.7akbkhvrmhu1) (gSLICr)

This is an external library: https://github.com/carlren/gSLICr

gSLICr is used to reduce the complexity of an image by grouping regions of similar color (these regions are called superpixels).

The configuration file is superpixel_filter.yaml
This configuration file is used by any detectors that utilize the SuperpixelFilter class, which is used to bridge gSLICr with the rest of the vision library.

These are the settings:

### gSLICr Settings

Filter Dimensions (input images will be resized to this, potentially increasing GPU load)
  
* gSLICr_width: 1920 (int)
* gSLICr_height: 1080 (int)

The range parameters significantly affect the filter's range. Split segments can be joined using the automatic filter.
  
* gSLICr_noSegs: 2000 (int) (**will segfault if set to 400**) - Higher means less distortions but higher change countours will be split
* gSLICr_spixelSize: 16 (int) - hint for gSLICr, it will not necessarily use this number
* gSLICr_sizeInsteadOfNumber: 0 (int: 0 or 1) - use a target size instead of number of pixels

Superpixel connectivity should always be set to true because only closed contours can result in detections. Altering the weight and iterations has not been tested and gSLICr documentation is lacking.
 
* gSLICr_cohWeight: 0.6 (double)
* gSLICr_noIters: 5 (int)
* gSLICr_doEnforceConnectivity: true (boolean)

### Performance

* gSLICr is the biggest load in the detection chain. (GPU heavy)

### Debug Image Topics

The superpixel filter is implemented as a utility class. Hence it does not have its own debug topics.

## [Automatic Filter](https://docs.google.com/document/d/14jtieKie-VdDFGFD33zAzqk0EjsogXXo8KFL7570wJk/edit)

There are two versions of the automatic filter, one for "color analysis" (old version / qualification detectors) and one for "shape analysis" (newer version / shape matcher). This documentation refers to the newer version.

The automatic filter is chained after superpixel image segmentation in order to group split superpixels and also to convert the image into a gray mask. Edge detection is run on this gray mask in order to find contours with OpenCV in a single pass.

Ideally, gSLICr should be set with a low enough number of segments that minimal contour separation occurs, then the automatic filter should be used with a medium number of bins such that it will fix separations but not aggressively.

### Automatic Filter Settings

Each detector has a .yaml file that configures the automatic filter for its specific goals.

The filter bins are the number of bins help determine the clusters that contain the same/similar color, a lower value means more colors end up in the same cluster. The black and white margins are how far away to uniformly threshold away from black and white, adding these simple threshold masks directly to the automatic filter's output mask.

* shape_detector_auto_filter_bins: 100 (unsigned char)
* shape_detector_auto_filter_black_margin: 30 (unsigned char)
* shape_detector_auto_filter_white_margin: 30 (unsigned char)

### Debug Image Topics

The automatic filter is implemented as a utility class. Hence it does not have its own debug topics.

### Performance

* The automatic filter is a relatively low load. (CPU heavy)

## [Qualification Detectors](https://docs.google.com/document/d/1GmmPfCVJnz3gBeASOMX9VwPkXvXWj9p3VMcPonEPcpA/edit#)

These are the detectors designed to detect the gate and post for the RoboSub 2018 qualification task. Significant work should not be put into these detectors as they are to be superceded by the shape matcher.

In the detector handler, their names are:
  qualification_gate_detector
  qualification_post_detector

## [Shape DB Creator](https://docs.google.com/document/d/1EHzh1h5DGaNLmPUfLZuOxqKJJGOFySl8--2vWqHwstQ/edit)

The shape DB creator is used to generate .sa files which are then fed into the shape matcher. DB files are plain text, so care should be taken to minimize the number of contours placed into the DB to avoid very large files.

The shape DB creator accepts a model (.3ds, etc.) on the "model" parameter. It will use this model along with the shape_db_creator.yaml file to render the model at varying angles, generating contours which can be later loaded to matching.

Note that the shape DB creator will not remove duplicate contours from generated DBs. For this reason it is up to the user to ensure that grabbed contours are mostly unique.

If an interrupt is sent to the DB creator during processing, the DB will stop rendering and will try to save what it has so far. Though, for larger DBs, it is likely that SIGTERM will be sent before saving finishes.

New DBs will always be written to "/au_vision/shape_dbs/newDB.sa". The DB creator will silently overwrite this file if it already exists, which makes guess and check easier when changing the DB creator config.

### Shape DB Creator Settings

There is only one shape DB creator .yaml file, so the if multiple configurations are saved under different names, they need to be renamed to "shape_db_creator.yaml" before running the DB creator.

The frame buffer dimensions are the render viewport size, as well as the window that contours are maximized within for area differencing (maintining aspect ratio). The frame buffer is divided into a grid, where the area a shape fills each grid square is stored, so that areas can be calculated much faster in a rough way by subtracting the area that two shapes fill a grid square, rather than render both shapes and do a pixel-wise subtraction. Since the frame buffer is divided into a grid, the square size must divide the frame buffer dimensions evenly.

* FrameBufferWidth: 1920 (int)
* FrameBufferHeight: 1080 (int)
* SquareSize: 12 (int)

The name string is the tag that will be added to the final matched ROI (not the parts in the case of multi-contour objects). The render distance is used for distance approximation. It should be an arbitrary value decided to be the the distance between the render camera and the model.

* Name: path (string)
* RenderDistance: 10.0 (double)

Setting Fps to 0 will make the renderer go as fast as possible. gSLICron is used to turn on gSLICr. ContourLinearizationEpsilon is used to reduce the number of points contours (especially arcs) for performance. A higher epsilon gives less points. The minimum contour area is used to cull contours that are too small. This is useful for objects that have small holes in them, for example.

* Fps: 0 (int)
* gSLICrOn: false (boolean)
* ContourLinearizationEpsilon: 3.0 (double)
* MinimumContourArea: 1500 (int)

The following parameters are used to change the initial model pose in case there is difficulty in setting up the actual model defaults, for example, if setting the origin accurately is difficult, or if the default orientation is undesired.

* RelativeTranslateX: 0 (double)
* RelativeTranslateY: 0 (double)
* RelativeTranslateZ: 0 (double)
* UniformScale: 1 (double)
* InitialRotX: 0 (double)
* InitialRotY: 0 (double)
* InitialRotZ: 0 (double)

These parameters decide what orientations to grab contours from. The start and end rotations on each axis cannot be the same. Any of these values can be negative except for the rotation step. The rotation step decides how often the renderer will grab contours within the passed axii ranges. It should not be set too low since this will greatly increase the load on the real time renderer in the shape matcher, since there will be many more potential shapes for each real time contour that need to have a fine area (pixel-wise) area difference taken. However, if the rotation step is set too high, the area difference tolerance in the shape matcher will need to be increased, potentially causing false positives.

Note that degrees are used over radians. This is for easier looping within the DB creator as well as to make debug output visuals more intuitive.

* StartRotateX: -20 (int)
* EndRotateX: 20 (int)
* StartRotateY: -20 (int)
* EndRotateY: 20 (int)
* StartRotateZ: 0 (int)
* EndRotateZ: 359 (int)
* RotateStep: 6 (int)

The shape DB creator does not use the automatic filter since DB results need to be very accurate. For this reason, simple thresholding is employed. There are three filter modes: range, include, exclude.

The filter count parameter indicates how many filters to read from the .yaml file. Filter parameter names should be zero-indexed, e.g.:
Filter_0_Mode: 
Filter_1_Mode:
Etc.

If the filter count is set less than the number of filters in the .yaml file, they will be ignored.

* FilterCount: 1 (int)

The range filter mode will only find contours within the passed CIE LAB color range.

* Filter_0_Mode: range (string)
* Filter_0_LowerL: 0 (unsigned char)
* Filter_0_LowerA: 138 (unsigned char)
* Filter_0_LowerB: 138 (unsigned char)
* Filter_0_UpperL: 255 (unsigned char)
* Filter_0_UpperA: 255 (unsigned char)
* Filter_0_UpperB: 255 (unsigned char)

The include and exclude filters have a different parameter format:

* Filter_0_Mode: include (string)
* Filter_0_L: 0 (unsigned char)
* Filter_0_A: 0 (unsigned char)
* Filter_0_B: 0 (unsigned char)
* Filter_0_Margin: 0 (unsigned char)

The margin will be added and subtracted uniformly from the CIE LAB target to create a filter range. If the mode is include, only the given target will be used for contour grabbing, and all other colors will be ignored. If the mode is exclude, only the given target will be ignored, and all other colors will be used (e.g. to exclude the background color easily).

Filters of all modes will be combined. Note that filters which fully contain each other are illegal and will cause an assertion.

### Debug Image Topics

* transformView
* contourView
* contourProcessedView

The tranform view topic will show the rendered model in full color. It is useful for ensuring that the model is rendering correctly. For example, if models are exported from blender as .3ds files, but the blender model contains multiple objects, only the first object will render. This topic makes this easy to catch.

The contour view will show the current pose as well as the pose ranges, allowing the user to estimate time remaining for generation. It will also indicate the maximum number of contours per frame, which acts as a sanity check since OpenSceneGraph has a tendency to mis-render. It also draws the found contours, their points, and their masks. This is the main window to observe.

The contour processed view will show contours that have been centered and maximized (maintaining aspect ratio) within the frame buffer dimensions. This topic will appear scrambled for multi-contour models.

### Performance

The shape DB creator can create a reasonably sized DB (~500 MB) within a few minutes. The bottleneck is the renderer's speed (OpenSceneGraph). 

## [Shape Matcher](https://docs.google.com/document/d/11veB_O6RGKTn0zjKASsya4Q-X121A7gLCWVsiSX9mG0/edit)

The shape matcher code is contained within the "shape_analysis" folders. It is the successor of the "color_analysis" code as it attempts to be as color invariant as possible, at the cost of requiring a more fine shape analyzing solution. The primary advantage of the shape matcher, over deep learning, is the ability to train a model in only a few minutes, meaning that our computer vision is very adaptive without requiring multiple specific detectors to be written and maintained; we can quickly get detections on new shapes simply by creating a 3D model from which to generate a DB. Too, improvements to one detector will affect all other detectors positively since there is a shared backend.

The shape matcher requires a .sa file to be generated by the shape DB creator for each detector that uses it as a backend. These .sa models should be placed into the "/au_vision/shape_dbs" folder and be referenced within the necessary .yaml file.

The backend is a implemented through a single function that takes a detector input image as well as some meta information about that detector. In this way, detectors using the shape matcher can easily get ROIs, but also can add extra code before or after the utility is called, for example, to cull ROIs that are in coming from water reflections, etc.

If shape matcher models (.sa) have been pre-generated, they will be stored on the team google drive: https://drive.google.com/open?id=1b25ZMJHWIixkcjgn22LsdK7p1ahZPo9e

### Shape Matcher Settings

Unlike gSLICr, the old automatic filter, and the shape DB creator, the shape matcher is configured on a per detector basis. For this reason, it is necessary to nest all shape matcher parameters within a namespace. This namespace should be named exactly the same as the detector name which is used by the detector handler to call the detector.

The database parameter is the DB name relative to "/au_vision/shape_dbs/".

The minimum contour area is the minimum area a contour must have to be considered for matching.

The contour linearization epsilon is how much to simplify contours (reduce points for performance at the cost of accuracy). Higher is more simple.

The area difference threshold is the maximum area difference a real time contour and a shape in the DB can have and still be considered associated.

The color difference threshold is the maximum color difference (on all channels) that a real time contour and a shape in the DB can have and still be considered associated.

The minimum rating is the minimum rating a group must have in order to be considered a match. This does not refer to the minimum rating of an individual shape in any way.

The auto filter parameters are described in the auto filter section. Note the boolean for turning off the auto filter histogram debug output topic. If this is set to true, a very large amount of processing time will be spent rendering the histogram image. This processing time goes up as the histogram bins goes up.

* path_detector: (namespace)
  * database: new_path.sa (string)
  * minimum_contour_area: 10000 (int)
  * contour_linearization_epsilon: 3.0 (double)
  * area_difference_thresh: 200000 (int)
  * color_difference_thresh: 1000 (int)
  * minimum_rating: 0.50 (double)
  * auto_filter_bins: 100 (unsigned char)
  * auto_filter_black_margin: 30 (unsigned char)
  * auto_filter_white_margin: 30 (unsigned char)
  * auto_filter_histogram: false (boolean)

### Debug Image Topics

* debug_GraySuperPixels
* debug_LineOverlay
* debug_AverageColors
* debug_InputStageContours
* debug_AutoFilterMask
* debug_AutoFilterHistogram
* debug_BinaryMaskWithHud
* debug_Edges

The gray super pixels topic will show gSLICr's native output, which is a mask with multiple grays, where each gray uniquely corresponds to a single superpixel.

The line overlay topic will show gSLICr's native wireframe overlay on the color input image. It is useful for ensuring that gSLICr is correctly outlining objects.

The average colors topic will show each superpixel colored with the average of all pixels within each respective superpixel.

The input stage contours topic will show found contours before they are culled. This is useful for tuning the minimum area threshold.

The auto filter shows a mask with multiple grays, where each gray corresponds to a cluster within the auto filter histogram.

The auto filter histogram topic will be empty if the auto filter histogram topic is not enabled. Otherwise it shows a 2D histogram with the AB components of the CIE LAB color space. Empty histogram squares are indicated with a smaller red square. Filled histogram squares are indicated with a smaller blue square, but their shade will also range from black to white, where white indicates a larger number of pixels.

The binary mask with HUD topic will output an image displaying the contours which make it past the culling function. Red contours indicate that they were clipped (did not pass the area / color difference threshold). Blue contours indicate that they were not clipped, but did not match within a group. Green contours indicate that they were matched within a group. Note that there is only ever one positively matched group, so multiple green contours can be considered associated.

The edges topic will output a binary mask of the auto filter mask's edges. This is the image that is passed to OpenCV's contour finding algorithm.