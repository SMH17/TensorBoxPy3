cmake_minimum_required(VERSION 3.8)
project(stitchwrapper)

set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(PythonLibs)

# Invoke cython

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stitch_wrapper.cpp
    COMMAND cython -o ${CMAKE_CURRENT_BINARY_DIR}/stitch_wrapper.cpp ${CMAKE_CURRENT_SOURCE_DIR}/stitch_wrapper.pyx
)

# Build Library

include_directories("." "./hungarian" ${PYTHON_INCLUDE_DIRS})
add_library(stitch_wrapper SHARED ${CMAKE_CURRENT_BINARY_DIR}/stitch_wrapper.cpp stitch_rects.cpp ./hungarian/hungarian.cpp)
target_link_libraries(stitch_wrapper ${PYTHON_LIBRARIES})

# Copy file to source tree

add_custom_target(copy_stitch_wrapper ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:stitch_wrapper> ${CMAKE_CURRENT_SOURCE_DIR}/stitch_wrapper.pyd
)
add_dependencies(copy_stitch_wrapper stitch_wrapper)