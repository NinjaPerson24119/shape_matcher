
cmake_minimum_required(VERSION 2.8.3)
project(au_vision)

#Enable C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#Optimize or add debug symbols
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_definitions(-O3)
else()
  add_definitions(-g)
endif()

# Define path of gSLICr folder here
find_path(GSLICR_PATH
	NAMES "README.md"
	HINTS "${CMAKE_CURRENT_SOURCE_DIR}/gSLICr/")
message(STATUS "gSLICr path dir = ${GSLICR_PATH}")

# Find CUDA
find_package(CUDA 9 REQUIRED)

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

  find_package(OpenGL REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS})

  cuda_add_library(shape_analysis
    src/shape_analysis/shape_analysis.cpp
    src/shape_analysis/shape_analysis_kernels.cu
    src/shape_analysis/shape_analysis_matcher.cpp
    src/shape_analysis/contour_renderer.cpp
    src/shape_analysis/auto_filter.cpp
  )

  add_dependencies(shape_analysis
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(shape_analysis
    gSLICr_lib
    gpu_util
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  #########################
  ##  Shape DB Creator   ##
  #########################

  find_package(OpenSceneGraph REQUIRED osgDB osgUtil osgViewer osgGA)
  include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})

  add_executable(shape_db_creator src/shape_analysis/shape_db_creator.cpp)

  add_dependencies(shape_db_creator
    ${PROJECT_NAME}_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(shape_db_creator
	  gpu_util
    shape_analysis
    ${OpenCV_LIBRARIES}
    ${OPENSCENEGRAPH_LIBRARIES}
    ${catkin_LIBRARIES}
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

  #############################
  ##      OpenCV (NO GPU)    ##
  #############################
  find_package(OpenCV REQUIRED)

  include_directories(
    ${OpenCV_INCLUDE_DIRS}
  )

endif()