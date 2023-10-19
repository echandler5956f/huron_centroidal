cmake_minimum_required(VERSION 3.0.2)
project(huron_centroidal)

find_package(catkin REQUIRED COMPONENTS message_generation rospy)

catkin_package(
  CATKIN_DEPENDS message_runtime
)