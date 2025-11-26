cmake_minimum_required( VERSION 2.8 )

project( GEOMETRY )

include_directories( "/usr/include/eigen3")
add_executable( geometry useGeometry.cpp)