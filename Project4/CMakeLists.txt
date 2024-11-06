cmake_minimum_required(VERSION 2.8)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

project( corner_detection )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include )

add_library(utils STATIC utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})

add_library( calibration STATIC calibration.cpp )
target_link_libraries( calibration ${OpenCV_LIBS} utils )

add_executable( corner_detection corner_detection.cpp )
target_link_libraries( corner_detection ${OpenCV_LIBS} calibration utils )

add_executable( harris_corner harris_corner.cpp )
target_link_libraries( harris_corner ${OpenCV_LIBS} utils )

# find_package(GLEW REQUIRED)
# find_package(glfw3 REQUIRED)
# find_package(Assimp REQUIRED)
# find_package(glm REQUIRED)

# Add executable target
# add_executable(test test.cpp)

# Link libraries
# target_link_libraries(test 
#     ${OpenCV_LIBS}
#     GLEW::GLEW
#     glfw
#     ${ASSIMP_LIBRARIES}
# )

# Include directories
# target_link_libraries(test 
#     ${OpenCV_LIBS}
#     GLEW::GLEW
#     glfw
#     ${ASSIMP_LIBRARIES}
#     GL
#     ${OPENGL_LIBRARIES}  # Link OpenGL
# )

# target_include_directories(test PRIVATE
#     ${OpenCV_INCLUDE_DIRS}
#     ${GLEW_INCLUDE_DIRS}
#     ${ASSIMP_INCLUDE_DIRS}
# )

# include_directories("/usr/include/glm")  # Adjust the path if GLM is located elsewhere

# find_package(OpenGL REQUIRED)
# find_package(GLUT REQUIRED) # Optional if using GLFW instead of GLUT

# add_executable(test test.cpp)

# Link libraries
# target_link_libraries(test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
