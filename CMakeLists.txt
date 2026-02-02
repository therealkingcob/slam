cmake_minimum_required(VERSION 3.10)
project(vo1 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(main
    main.cpp
    ORB_pattern.cpp
)

target_link_libraries(main
    PRIVATE
    ${OpenCV_LIBS}
    Ceres::ceres
)
