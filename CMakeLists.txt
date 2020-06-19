cmake_minimum_required(VERSION 3.10)
project(eyetrack_mouse)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

add_subdirectory(src)

