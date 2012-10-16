project(CAMGRASSHOPPER)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/camGrasshopperConfig.txt ${CMAKE_BINARY_DIR}/bin/camGrasshopperConfig.txt)

add_library(camGrasshopper MODULE camGrasshopper.cc grasshopper.cc)
set(FLYCAPTURE_INCLUDE_DIR "/usr/include/flycapture")
include_directories(SYSTEM ${FLYCAPTURE_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
target_link_libraries(camGrasshopper bvs flycapture opencv_core opencv_imgproc)
