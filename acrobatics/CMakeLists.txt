# Source/Include files
include_directories(".")
include_directories("include/" ) # acrobatics includes

include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE sources "src/*.cpp") # acrobatics library cpp files

find_package(casadi)
find_package(pinocchio)
find_package (Eigen3 REQUIRED NO_MODULE)

# Library
add_library(biqu-acrobatics SHARED ${sources}) # produce a library used by interface
target_link_libraries(biqu-acrobatics casadi pinocchio Eigen3::Eigen)

add_executable(test src/test.cpp)
target_link_libraries(test biqu-acrobatics)