cmake_minimum_required(VERSION 3.7)
project(code)

find_package(OpenCV 3.4.2 REQUIRED)
find_package(PCL 1.8.0 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp feature3d.cpp feature3d.h feature2d.cpp feature2d.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES})