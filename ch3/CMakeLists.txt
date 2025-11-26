cmake_minimum_required( VERSION 2.8 )
project( MATRIX )

include_directories( "/usr/include/eigen3")
add_executable( matrix eigenMatrix.cpp)
