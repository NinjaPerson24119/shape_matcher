Shape DB Creator
The shape DB creator is used to generate .sa files which are then fed into the shape matcher. DB files are plain text, so care should be taken to minimize the number of contours placed into the DB to avoid very large files.
The shape DB creator accepts a model (.3ds, etc.) on the "model" parameter. It will use this model along with the shape_db_creator.yaml file to render the model at varying angles, generating contours which can be later loaded to matching.
Note that the shape DB creator will not remove duplicate contours from generated DBs. For this reason it is up to the user to ensure that grabbed contours are mostly unique.
If an interrupt is sent to the DB creator during processing, the DB will stop rendering and will try to save what it has so far. Though, for larger DBs, it is likely that SIGTERM will be sent before saving finishes.
New DBs will always be written to "/au_vision/shape_dbs/newDB.sa". The DB creator will silently overwrite this file if it already exists, which makes guess and check easier when changing the DB creator config.
Shape DB Creator Settings
There is only one shape DB creator .yaml file, so the if multiple configurations are saved under different names, they need to be renamed to "shape_db_creator.yaml" before running the DB creator.
The frame buffer dimensions are the render viewport size, as well as the window that contours are maximized within for area differencing (maintining aspect ratio). The frame buffer is divided into a grid, where the area a shape fills each grid square is stored, so that areas can be calculated much faster in a rough way by subtracting the area that two shapes fill a grid square, rather than render both shapes and do a pixel-wise subtraction. Since the frame buffer is divided into a grid, the square size must divide the frame buffer dimensions evenly.
FrameBufferWidth: 1920 (int)
FrameBufferHeight: 1080 (int)
SquareSize: 12 (int)
The name string is the tag that will be added to the final matched ROI (not the parts in the case of multi-contour objects). The render distance is used for distance approximation. It should be an arbitrary value decided to be the the distance between the render camera and the model.
Name: path (string)
RenderDistance: 10.0 (double)
The framerate parameter is used to cap the render speed for debugging purposes, but setting it to 0 will make the renderer go as fast as possible. Since gSLICr is very slow, there is an option to turn it off since most simple objects would not need it anyway. The contour linearization epsilon is used to reduce the number of points contours (especially arcs) for performance reasons. A higher epsilon will result in less points. The minimum contour area is used to cull contours that are too small. This is useful for objects that have small holes in them, for example.
Fps: 0 (int)
gSLICrOn: false (boolean)
ContourLinearizationEpsilon: 3.0 (double)
MinimumContourArea: 1500 (int)
The following parameters are used to change the initial model pose in case there is difficulty in setting up the actual model defaults, for example, if setting the origin accurately is difficult, or if the default orientation is undesired.
RelativeTranslateX: 0 (double)
RelativeTranslateY: 0 (double)
RelativeTranslateZ: 0 (double)
UniformScale: 1 (double)
InitialRotX: 0 (double)
InitialRotY: 0 (double)
InitialRotZ: 0 (double)
These parameters decide what orientations to grab contours from. The start and end rotations on each axis cannot be the same. Any of these values can be negative except for the rotation step. The rotation step decides how often the renderer will grab contours within the passed axii ranges. It should not be set too low since this will greatly increase the load on the real time renderer in the shape matcher, since there will be many more potential shapes for each real time contour that need to have a fine area (pixel-wise) area difference taken. However, if the rotation step is set too high, the area difference tolerance in the shape matcher will need to be increased, potentially causing false positives.
Note that degrees are used over radians. This is for easier looping within the DB creator as well as to make debug output visuals more intuitive.
StartRotateX: -20 (int)
EndRotateX: 20 (int)
StartRotateY: -20 (int)
EndRotateY: 20 (int)
StartRotateZ: 0 (int)
EndRotateZ: 359 (int)
RotateStep: 6 (int)
The shape DB creator does not use the automatic filter since DB results need to be very accurate. For this reason, simple thresholding is employed. There are three filter modes: range, include, exclude.
The filter count parameter indicates how many filters to read from the .yaml file. Filter parameter names should be zero-indexed, e.g.: Filter_0_Mode: Filter_1_Mode: Etc.
If the filter count is set less than the number of filters in the .yaml file, they will be ignored.
FilterCount: 1 (int)
The range filter mode will only find contours within the passed CIE LAB color range.
Filter_0_Mode: range (string)
Filter_0_LowerL: 0 (unsigned char)
Filter_0_LowerA: 138 (unsigned char)
Filter_0_LowerB: 138 (unsigned char)
Filter_0_UpperL: 255 (unsigned char)
Filter_0_UpperA: 255 (unsigned char)
Filter_0_UpperB: 255 (unsigned char)
The include and exclude filters have a different parameter format:
Filter_0_Mode: include (string)
Filter_0_L: 0 (unsigned char)
Filter_0_A: 0 (unsigned char)
Filter_0_B: 0 (unsigned char)
Filter_0_Margin: 0 (unsigned char)
The margin will be added and subtracted uniformly from the CIE LAB target to create a filter range. If the mode is include, only the given target will be used for contour grabbing, and all other colors will be ignored. If the mode is exclude, only the given target will be ignored, and all other colors will be used (e.g. to exclude the background color easily).
Filters of all modes will be combined. Note that filters which fully contain each other are illegal and will cause an assertion.
Debug Image Topics
transformView
contourView
contourProcessedView
The tranform view topic will show the rendered model in full color. It is useful for ensuring that the model is rendering correctly. For example, if models are exported from blender as .3ds files, but the blender model contains multiple objects, only the first object will render. This topic makes this easy to catch.
The contour view will show the current pose as well as the pose ranges, allowing the user to estimate time remaining for generation. It will also indicate the maximum number of contours per frame, which acts as a sanity check since OpenSceneGraph has a tendency to mis-render. It also draws the found contours, their points, and their masks. This is the main window to observe.
The contour processed view will show contours that have been centered and maximized (maintaining aspect ratio) within the frame buffer dimensions. This topic will appear scrambled for multi-contour models.
Performance
The shape DB creator can create a reasonably sized DB (~500 MB) within a few minutes. The bottleneck is the renderer's speed (OpenSceneGraph).


