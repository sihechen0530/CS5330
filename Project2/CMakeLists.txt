cmake_minimum_required(VERSION 2.8)

project( imageMatch )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(csv_util STATIC csv_util.cpp)
add_library(utils STATIC utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})

add_library(feature_extractor STATIC feature_extractor.cpp)
target_link_libraries(feature_extractor ${OpenCV_LIBS} utils)

add_library(metric_calculator STATIC metric_calculator.cpp)
target_link_libraries(metric_calculator ${OpenCV_LIBS} utils)

add_library(vector_database STATIC vector_database.cpp)
target_link_libraries(vector_database utils csv_util feature_extractor metric_calculator)

add_executable(imageMatch imageMatch.cpp vector_database.cpp)
target_link_libraries( imageMatch ${OpenCV_LIBS} vector_database utils)
