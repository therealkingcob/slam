cmake_minimum_required( VERSION 2.8 )
project( HELLO )
add_executable( hello hello.cpp)
add_library( hello libHelloSLAM.cpp )
