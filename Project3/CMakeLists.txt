cmake_minimum_required(VERSION 2.8)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

project( imageMatch )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(utils STATIC utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})

add_library(segmentation STATIC segmentation.cpp)
target_link_libraries(segmentation ${OpenCV_LIBS} utils)

add_library(feature STATIC feature.cpp)
target_link_libraries(feature ${OpenCV_LIBS} utils)

add_executable(2DRecog 2DRecog.cpp)
target_link_libraries( 2DRecog ${OpenCV_LIBS} segmentation feature utils)
