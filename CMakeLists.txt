project(CAMGRASSHOPPER)

# FlyCapture
set(FLYCAPTURE_INCLUDE_DIR "/usr/include/flycapture")
include_directories(SYSTEM ${FLYCAPTURE_INCLUDE_DIR}) # surpress flycapture warnings

# OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# OpenCL
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake") # add cmake directory for FindOpenCL.cmake
find_package(OpenCL)
if (OPENCL_FOUND)
	message(STATUS "OpenCL found!")
	add_definitions(-D_WITH_OPENCL)
	include_directories(${OPENCL_INCLUDE_DIRS})
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${OPENCL_LIBRARIES})
	include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS}) # surpress OpenCL warnings
	set(OpenCL_LIB OpenCL)
	# convert the OpenCL kernel file to a const char array
	# const char clProgramCode[] = yuv422toRgb.cl --> yuv422toRgb.h
	file(READ "yuv422toRgb.cl" CL_KERNEL)
	set(CL_KERNEL_HEADER "const char clProgramCode[]=\"${CL_KERNEL}")
	STRING(REGEX REPLACE "\n" "\"\n\"\\\\n" CL_KERNEL_HEADER "${CL_KERNEL_HEADER}")
	set (CL_KERNEL_HEADER "${CL_KERNEL_HEADER}\\0\";") # add ending character and semicolon
	set (CL_KERNEL_HEADER "\#ifndef _YUV422_TO_RGB_H_\n\#define _YUV422_TO_RGB_H_\n${CL_KERNEL_HEADER}\n\#endif") # add preprocessor directives
	file(WRITE "yuv422toRgb.h" "${CL_KERNEL_HEADER}")
endif (OPENCL_FOUND)

# BVS module camGrasshopper
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/camGrasshopper.conf ${CMAKE_BINARY_DIR}/bin/camGrasshopper.conf)
add_library(camGrasshopper MODULE camGrasshopper.cc grasshopper.cc)
target_link_libraries(camGrasshopper bvs flycapture opencv_core opencv_imgproc ${OpenCL_LIB})

# Grasshopper standalone demo
add_executable(grasshopper-demo grasshopper.cc)
set_target_properties(grasshopper-demo PROPERTIES COMPILE_FLAGS "-D_STANDALONE")
target_link_libraries(grasshopper-demo flycapture opencv_core opencv_highgui opencv_imgproc ${OpenCL_LIB})
