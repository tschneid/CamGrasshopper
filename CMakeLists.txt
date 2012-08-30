project(CAMGRASSHOPPER)

add_library(camGrasshopper MODULE camGrasshopper.cc grasshopper.cc)
set(FLYCAPTURE_INCLUDE_DIR "/usr/include/flycapture")
include_directories(${FLYCAPTURE_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
target_link_libraries(camGrasshopper bvs flycapture opencv_core opencv_imgproc)
