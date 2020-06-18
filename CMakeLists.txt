cmake_minimum_required(VERSION 3.10)

project(eyetrack_mouse)

include_directories(include)
include_directories(data)

find_package(OpenCV REQUIRED)
add_subdirectory(/opt/dlib dlib)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cpp")

add_executable(eyetrack_mouse ${SOURCES})

target_link_libraries(eyetrack_mouse PRIVATE "opencv_dnn_objdetect")
target_link_libraries(eyetrack_mouse PRIVATE dlib::dlib)
