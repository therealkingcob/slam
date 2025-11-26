cmake_minimum_required( VERSION 2.8 )
project( COORDINATE )
include_directories( "/usr/include/eigen3" )
add_executable(coordinate coordinateTransform.cpp)