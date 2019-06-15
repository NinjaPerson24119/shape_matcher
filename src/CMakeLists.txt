
cmake_minimum_required(VERSION 3.10.2)
project(au_vision)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

#Optimize or add debug symbols
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_definitions(-O3)
else()
  add_definitions(-g)
endif()

# Define path of darknet folder here.
find_path(DARKNET_PATH
	NAMES "README.md"
	HINTS "${CMAKE_CURRENT_SOURCE_DIR}/darknet/")
message(STATUS "Darknet path dir = ${DARKNET_PATH}")


# Define path of gSLICr folder here
find_path(GSLICR_PATH
	NAMES "README.md"
	HINTS "${CMAKE_CURRENT_SOURCE_DIR}/gSLICr/")
message(STATUS "gSLICr path dir = ${GSLICR_PATH}")

# Find CUDA
find_package(CUDA 10)

if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/check_cuda)
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
    message(STATUS "CUDA lib64 Path: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH}")
    add_definitions(-DGPU)

    #Get compute version as numerical string
    string(REGEX REPLACE "[^0-9.]" "" CUDA_COMPUTE_VERSION ${ARCH})

    #Add library path to linker
    link_directories(
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    )
  else()
    message(WARNING ${ARCH})
  endif()
endif()

if (NOT CUDA_FOUND OR NOT ${CUDA_SUCCESS})
  list(APPEND LIBRARIES "m")
  # Set CUDA include directories as empty
  set(CUDA_INCLUDE_DIRS)
endif()

# Find required packages
find_package(Boost REQUIRED COMPONENTS thread)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nodelet
  cv_bridge
  image_transport
  actionlib
  dynamic_reconfigure
  message_generation
  std_msgs
  std_srvs
  sensor_msgs
  actionlib_msgs
  au_core
  ar_track_alvar_msgs
)

catkin_python_setup()

add_service_files(
  DIRECTORY srv
  FILES DetectObjects.srv
)

add_action_files(
  DIRECTORY action
  FILES Detection.action
)

generate_dynamic_reconfigure_options(
  cfg/QualificationFilter.cfg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  std_srvs
  actionlib_msgs
  au_core
)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES detectors
   CATKIN_DEPENDS message_runtime std_msgs sensor_msgs std_srvs actionlib_msgs au_core
)

###########
## Build ##
###########

if(${CUDA_SUCCESS})
  #Remove the references to ROS OpenCV that catkin automatically injects
  string(REGEX REPLACE ";?\\/[a-z/_]+opencv[^;]+" "" catkin_LIBRARIES "${catkin_LIBRARIES}")
  string(REGEX REPLACE ";?\\/[a-z/_]+opencv[^;]+" "" catkin_INCLUDE_DIRS "${catkin_INCLUDE_DIRS}")
endif()

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

#####################
##       GPU       ##
#####################

# file(GLOB ...) should not be used here as it prevents CMake from detecting changes,
# and thus rebuilding, when any of the files are modified or removed
set(DARKNET_SOURCE
  ${DARKNET_PATH}/src/activation_layer.c
  ${DARKNET_PATH}/src/activations.c
  ${DARKNET_PATH}/src/avgpool_layer.c
  ${DARKNET_PATH}/src/batchnorm_layer.c
  ${DARKNET_PATH}/src/blas.c
  ${DARKNET_PATH}/src/box.c
  ${DARKNET_PATH}/src/col2im.c
  ${DARKNET_PATH}/src/connected_layer.c
  ${DARKNET_PATH}/src/convolutional_layer.c
  ${DARKNET_PATH}/src/cost_layer.c
  ${DARKNET_PATH}/src/crnn_layer.c
  ${DARKNET_PATH}/src/crop_layer.c
  ${DARKNET_PATH}/src/cuda.c
  ${DARKNET_PATH}/src/data.c
  ${DARKNET_PATH}/src/deconvolutional_layer.c
  ${DARKNET_PATH}/src/demo.c
  ${DARKNET_PATH}/src/detection_layer.c
  ${DARKNET_PATH}/src/detector.c
  ${DARKNET_PATH}/src/dropout_layer.c
  ${DARKNET_PATH}/src/gemm.c
  ${DARKNET_PATH}/src/gru_layer.c
  ${DARKNET_PATH}/src/http_stream.cpp
  ${DARKNET_PATH}/src/im2col.c
  ${DARKNET_PATH}/src/image.c
  ${DARKNET_PATH}/src/layer.c
  ${DARKNET_PATH}/src/list.c
  ${DARKNET_PATH}/src/local_layer.c
  ${DARKNET_PATH}/src/matrix.c
  ${DARKNET_PATH}/src/maxpool_layer.c
  ${DARKNET_PATH}/src/network.c
  ${DARKNET_PATH}/src/normalization_layer.c
  ${DARKNET_PATH}/src/option_list.c
  ${DARKNET_PATH}/src/parser.c
  ${DARKNET_PATH}/src/region_layer.c
  ${DARKNET_PATH}/src/reorg_layer.c
  ${DARKNET_PATH}/src/reorg_old_layer.c
  ${DARKNET_PATH}/src/rnn_layer.c
  ${DARKNET_PATH}/src/route_layer.c
  ${DARKNET_PATH}/src/shortcut_layer.c
  ${DARKNET_PATH}/src/softmax_layer.c
  ${DARKNET_PATH}/src/tree.c
  ${DARKNET_PATH}/src/upsample_layer.c
  ${DARKNET_PATH}/src/utils.c
  ${DARKNET_PATH}/src/yolo_layer.c
)

if(${CUDA_SUCCESS})

  ##########################
  ##      OpenCV (GPU)    ##
  ##########################

  #Get OpenCV if need be
  set(OPENCV_GPU_VERSION "3.3.1")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv-${OPENCV_GPU_VERSION}")
    #Download with wget
    message(STATUS "Downloading OpenCV ${OPENCV_GPU_VERSION}; will compile for GPU")
    execute_process(
      COMMAND wget --continue "https://github.com/opencv/opencv/archive/${OPENCV_GPU_VERSION}.tar.gz"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE wget_retcode
    )
    if(NOT "${wget_retcode}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error downloading opencv-${OPENCV_GPU_VERSION}")
    endif()

    #Extract archive
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_SOURCE_DIR}/${OPENCV_GPU_VERSION}.tar.gz"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    #Delete archive
    execute_process(
      COMMAND rm "${CMAKE_CURRENT_SOURCE_DIR}/${OPENCV_GPU_VERSION}.tar.gz"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()

  #Ensure that OpenCV's build dir exists
  set(OPENCV_GPU_BUILD_DIR "${CMAKE_BINARY_DIR}/../build_external/opencv-${OPENCV_GPU_VERSION}")
  if(NOT EXISTS ${OPENCV_GPU_BUILD_DIR} OR NOT EXISTS ${CATKIN_DEVEL_PREFIX}/lib/libopencv_core.so.${OPENCV_GPU_VERSION})
    execute_process(
      COMMAND mkdir -p ${OPENCV_GPU_BUILD_DIR}
    )

    #Configure OpenCV
    execute_process(
      COMMAND ${CMAKE_COMMAND}
      -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -D CMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
      -D CUDA_HOST_COMPILER=/usr/bin/g++
      -D BUILD_EXAMPLES=OFF
      -D BUILD_DOCS=OFF
      -D BUILD_PERF_TESTS=OFF
      -D BUILD_TESTS=OFF
      -D WITH_TBB=ON
      -D WITH_OPENMP=ON
      -D WITH_IPP=OFF
      -D WITH_NVCUVID=OFF
      -D WITH_CSTRIPES=OFF
      -D WITH_OPENCL=OFF
      -D ENABLE_FAST_MATH=ON
      -D WITH_CUDA=ON
      -D CUDA_ARCH_BIN=${CUDA_COMPUTE_VERSION}
      -D CUDA_ARCH_PTX=""
      -D CUDA_FAST_MATH=ON
      -D WITH_CUBLAS=ON
      -D WITH_OPENGL=ON
      -D CUDA_NVCC_FLAGS="--expt-relaxed-constexpr"
      "${CMAKE_CURRENT_SOURCE_DIR}/opencv-${OPENCV_GPU_VERSION}"
      WORKING_DIRECTORY ${OPENCV_GPU_BUILD_DIR}
    )

    #Build OpenCV
    include(ProcessorCount)
    ProcessorCount(CORES)
    if(CORES EQUAL 0)
      set(CORES 1)
    endif()

    execute_process(
      COMMAND make -j${CORES}
      WORKING_DIRECTORY ${OPENCV_GPU_BUILD_DIR}
    )

    execute_process(
      COMMAND make install
      WORKING_DIRECTORY ${OPENCV_GPU_BUILD_DIR}
    )
  endif() # end of build OpenCV

  #Setup OpenCV vars, unnecessary to set include dirs, since OpenCV is installed to /devel
  #Grep lib names
  execute_process(
    COMMAND bash "-c" "ls ${CATKIN_DEVEL_PREFIX}/lib | grep libopencv | grep -P \"lib(\\w+)\\.so(?!\\.)\""
    OUTPUT_VARIABLE OPENCV_GPU_GREP_LIBS
  )
  string(REGEX REPLACE "\n" ";" OPENCV_GPU_GREP_LIBS "${OPENCV_GPU_GREP_LIBS}")

  #Let CMake find them
  foreach(TEMP_LIB_NAME ${OPENCV_GPU_GREP_LIBS})
    set(OPENCV_GPU_LIBS ${OPENCV_GPU_LIBS} "${CATKIN_DEVEL_PREFIX}/lib/${TEMP_LIB_NAME}")
  endforeach(TEMP_LIB_NAME)
  set(OpenCV_LIBRARIES ${OPENCV_GPU_LIBS})

  #####################
  ##      gSLICr     ##
  #####################

  set(GSLICR_SRC
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_core_engine.h
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_seg_engine.h
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_core_engine.cpp
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_seg_engine.cpp
    ${GSLICR_PATH}/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
    ${GSLICR_PATH}/gSLICr_Lib/objects/gSLICr_settings.h
    ${GSLICR_PATH}/gSLICr_Lib/objects/gSLICr_spixel_info.h
    ${GSLICR_PATH}/gSLICr_Lib/gSLICr_defines.h
    ${GSLICR_PATH}/gSLICr_Lib/gSLICr.h
  )

  include_directories(
    ${GSLICR_PATH}
  )

  cuda_add_library(gSLICr_lib
  			${GSLICR_SRC}
  			${GSLICR_PATH}/NVTimer.h
  )
  target_link_libraries(gSLICr_lib ${CUDA_LIBRARIES})

  #################################
  ##           GPU Util          ##
  #################################

  cuda_add_library(gpu_util
    src/shape_analysis/gpu_util.cpp
    src/shape_analysis/gpu_util_kernels.cu
    src/shape_analysis/superpixel_filter.cpp
    src/shape_analysis/superpixel_filter_kernels.cu
  )

  add_dependencies(gpu_util
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(gpu_util
	  gSLICr_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  #########################
  ##    Shape Analysis   ##
  #########################

  cuda_add_library(shape_analysis
    src/shape_analysis/shape_analysis.cpp
    src/shape_analysis/shape_analysis_kernels.cu
    src/shape_analysis/shape_analysis_matcher.cpp
    src/shape_analysis/auto_filter.cpp
  )

  add_dependencies(shape_analysis
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(shape_analysis
    gSLICr_lib
    gpu_util
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  #########################
  ##  Shape DB Creator   ##
  #########################
  find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgViewer osgGA osg)
  include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
  
  find_package(OpenGL REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS})

  add_executable(shape_db_creator src/shape_analysis/shape_db_creator.cpp)

  add_dependencies(shape_db_creator
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(shape_db_creator
	  gpu_util
    shape_analysis
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OPENSCENEGRAPH_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  #####################################
  ##  Color Analysis / Qualification ##
  #####################################

  ## Create color_analysis library
  add_library(color_analysis
    src/color_analysis/color_analysis.cpp
    src/color_analysis/qualification_gate_detector.cpp
    src/color_analysis/qualification_post_detector.cpp
    src/color_analysis/auto_filter_old.cpp
  )

  add_dependencies( color_analysis
    au_core_generate_messages_cpp
    ${PROJECT_NAME}_gencfg
    ${PROJECT_NAME}_generate_messages_cpp
  )

  target_link_libraries(color_analysis
    gpu_util
    shape_analysis
    ${catkin_LIBRARIES}
  )

  set(DETECTORS_LINK_LIBRARIES
    ${DETECTORS_LINK_LIBRARIES}
    color_analysis
  )

  set(DETECTOR_NODES_LINK_LIBRARIES
    ${DETECTOR_NODES_LINK_LIBRARIES}
    color_analysis
  )

  #####################
  ## Shape Detectors ##
  #####################

  add_library(shape_detectors
    src/shape_analysis/shape_analysis_detector_common.cpp
  )

  add_dependencies(shape_detectors
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(shape_detectors
	  gpu_util
    shape_analysis
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  set(DETECTORS_LINK_LIBRARIES
    ${DETECTORS_LINK_LIBRARIES}
    shape_detectors
  )

  set(DETECTOR_NODES_LINK_LIBRARIES
    ${DETECTOR_NODES_LINK_LIBRARIES}
    shape_detectors
  )
  ###################
  ## Yolo Detector ##
  ###################

  set(DARKNET_KERNELS
    ${DARKNET_PATH}/src/activation_kernels.cu
    ${DARKNET_PATH}/src/avgpool_layer_kernels.cu
    ${DARKNET_PATH}/src/blas_kernels.cu
    ${DARKNET_PATH}/src/col2im_kernels.cu
    ${DARKNET_PATH}/src/convolutional_kernels.cu
    ${DARKNET_PATH}/src/crop_layer_kernels.cu
    ${DARKNET_PATH}/src/deconvolutional_kernels.cu
    ${DARKNET_PATH}/src/dropout_layer_kernels.cu
    ${DARKNET_PATH}/src/im2col_kernels.cu
    ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
    ${DARKNET_PATH}/src/network_kernels.cu
  )

  include_directories(
    ${DARKNET_PATH}/src
    ${DARKNET_PATH}/include
  )

  add_library(darknet_util
    src/yolo/darknet_util.cpp
  )
  # create yolo detector library
  cuda_add_library(yolo_detector
    src/detector.cpp
    src/yolo/yolo_detector.cpp
    ${DARKNET_SOURCE}
    ${DARKNET_KERNELS}
  )

  add_dependencies(yolo_detector
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )

  target_link_libraries(yolo_detector
    m
    pthread
    stdc++
    cuda
    cudart
    cublas
    curand
    darknet_util
    yaml-cpp
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  set(DETECTORS_LINK_LIBRARIES
    ${DETECTORS_LINK_LIBRARIES}
    yolo_detector
  )
else()
  message(STATUS "gSLICr will not be built - CUDA was not found")
  message(STATUS "Slope detector will not be built - CUDA was not found")

  #############################
  ##      OpenCV (NO GPU)    ##
  #############################
  find_package(OpenCV REQUIRED)

  include_directories(
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

#####################
##   Vision Util   ##
#####################

## Create vision_util library
add_library(vision_util
  src/vision_bridge.cpp
)
add_dependencies(vision_util
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(vision_util
  ${catkin_LIBRARIES}
)

#####################
##    Detectors    ##
#####################

# create detector library
add_library( detectors
  src/detector.cpp
  src/detector_factory.cpp
)

add_dependencies( detectors
  ${PROJECT_NAME}_gencfg
  ${PROJECT_NAME}_generate_messages_cpp
  ${catkin_EXPORTED_TARGETS}
)

#Use indirection to allow for conditional libraries
set(DETECTORS_LINK_LIBRARIES
  ${DETECTORS_LINK_LIBRARIES}
  vision_util
  ${catkin_LIBRARIES}
)

target_link_libraries( detectors
  ${DETECTORS_LINK_LIBRARIES}
)

###################
## Ar_tag Detector ##
###################
add_executable(artag_detector src/artag_detector.cpp)
add_dependencies(artag_detector
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(artag_detector
  ${catkin_LIBRARIES}
)

####################
## Detector Nodes ##
####################

#Avoid repetition and use indirection to allow for conditional libraries
set(DETECTOR_NODES_LINK_LIBRARIES
  ${DETECTOR_NODES_LINK_LIBRARIES}
  detectors
  ${catkin_LIBRARIES}
)

## Create Detector Handler
add_executable( detector_handler
  src/detector_handler/detector_handler.cpp
  src/detector_handler/detector_manager.cpp
  src/detector_handler/detector_handler_node.cpp
)
set_target_properties(detector_handler
        PROPERTIES COMPILE_FLAGS "-std=c++14")
add_dependencies( detector_handler
  std_msgs_generate_messages_cpp
  actionlib_msgs_generate_messages_cpp
  au_core_generate_messages_cpp
  ${PROJECT_NAME}_generate_messages_cpp
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries( detector_handler
  ${DETECTOR_NODES_LINK_LIBRARIES}
)

#########################
##     Mock Camera     ##
#########################

add_executable(mock_camera src/mock_camera.cpp)
add_dependencies(mock_camera
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(mock_camera
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
