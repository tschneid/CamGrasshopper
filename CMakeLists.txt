project(CAMGRASSHOPPER)

# convert the OpenCL kernel file to a const char array
# const char clProgramCode[] = yuv422toRgb.cl --> yuv422toRgb.h
file(READ "yuv422toRgb.cl" CL_KERNEL)
set(CL_KERNEL_HEADER "const char clProgramCode[]=\"${CL_KERNEL}")
STRING(REGEX REPLACE "\n" "\"\n\"\\\\n" CL_KERNEL_HEADER "${CL_KERNEL_HEADER}")
set (CL_KERNEL_HEADER "${CL_KERNEL_HEADER}\\0\";") # add ending character and semicolon
set (CL_KERNEL_HEADER "\#ifndef _YUV422_TO_RGB_H_\n\#define _YUV422_TO_RGB_H_\n${CL_KERNEL_HEADER}\n\#endif") # add preprocessor directives
file(WRITE "yuv422toRgb.h" "${CL_KERNEL_HEADER}")

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/camGrasshopper.conf ${CMAKE_BINARY_DIR}/bin/camGrasshopper.conf)

add_library(camGrasshopper MODULE camGrasshopper.cc grasshopper.cc)
set(FLYCAPTURE_INCLUDE_DIR "/usr/include/flycapture")
include_directories(SYSTEM ${FLYCAPTURE_INCLUDE_DIR}) # surpress flycapture warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# OpenCL
add_definitions(-D_WITH_OPENCL)
set(OPENCL_INCLUDE_DIR "/cvhci/distlibs/dists/ubuntu-12.04-amd64/cuda/include/")
set(OPENCL_LIB_DIR "/cvhci/distlibs/dists/ubuntu-12.04-amd64/cuda/lib64/")
include_directories(${OPENCL_INCLUDE_DIR}) # include dir
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${OPENCL_LIB_DIR}) # lib dir
include_directories(SYSTEM ${OPENCL_INCLUDE_DIR}) # surpress OpenCL warnings

target_link_libraries(camGrasshopper bvs flycapture opencv_core opencv_imgproc OpenCL)
