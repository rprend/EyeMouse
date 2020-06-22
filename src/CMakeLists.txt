cmake_minimum_required(VERSION 3.10)
project(eyetrack_src)

find_package(OpenCV REQUIRED)

set(SRC_SOURCE
    FaceEyeDetector.cpp
    FaceEyeDetector.h
    main.cpp
    camux/Eye.h
    camux/Face.cpp
    camux/Face.h
    camux/geometry.cpp
    camux/geometry.hpp
    )

add_subdirectory(/opt/dlib dlib)

add_executable(eye_mouse ${SRC_SOURCE})
target_link_libraries(eye_mouse ${OpenCV_LIBS} dlib::dlib)