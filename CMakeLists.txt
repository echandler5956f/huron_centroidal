cmake_minimum_required(VERSION 3.10.2)
project(BiQu Acrobatics)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 11)

find_package(casadi REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS unit_test_framework)

add_subdirectory(src)