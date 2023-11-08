cmake_minimum_required(VERSION 3.10.2)
project(test_env)

find_package(casadi)
find_package(pinocchio)
find_package (Eigen3 REQUIRED NO_MODULE)
FIND_PACKAGE(Boost COMPONENTS unit_test_framework)

add_executable(full_demo full_demo.cpp)
target_link_libraries(full_demo casadi pinocchio Eigen3::Eigen Boost::unit_test_framework)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

install(TARGETS casadi_demo pinocchio_demo full_demo RUNTIME DESTINATION . LIBRARY DESTINATION .)