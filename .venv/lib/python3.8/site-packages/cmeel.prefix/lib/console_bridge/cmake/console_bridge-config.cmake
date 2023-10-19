if (console_bridge_CONFIG_INCLUDED)
  return()
endif()
set(console_bridge_CONFIG_INCLUDED TRUE)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was console_bridge-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

####################################################################################

set(console_bridge_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")

include("${CMAKE_CURRENT_LIST_DIR}/console_bridge-targets.cmake")
set(console_bridge_LIBRARIES console_bridge::console_bridge)
set(console_bridge_TARGETS console_bridge::console_bridge)
