#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "urdfdom::urdfdom_model" for configuration "Release"
set_property(TARGET urdfdom::urdfdom_model APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(urdfdom::urdfdom_model PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liburdfdom_model.so.3.0"
  IMPORTED_SONAME_RELEASE "liburdfdom_model.so.3.0"
  )

list(APPEND _cmake_import_check_targets urdfdom::urdfdom_model )
list(APPEND _cmake_import_check_files_for_urdfdom::urdfdom_model "${_IMPORT_PREFIX}/lib/liburdfdom_model.so.3.0" )

# Import target "urdfdom::urdfdom_world" for configuration "Release"
set_property(TARGET urdfdom::urdfdom_world APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(urdfdom::urdfdom_world PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liburdfdom_world.so.3.0"
  IMPORTED_SONAME_RELEASE "liburdfdom_world.so.3.0"
  )

list(APPEND _cmake_import_check_targets urdfdom::urdfdom_world )
list(APPEND _cmake_import_check_files_for_urdfdom::urdfdom_world "${_IMPORT_PREFIX}/lib/liburdfdom_world.so.3.0" )

# Import target "urdfdom::urdfdom_sensor" for configuration "Release"
set_property(TARGET urdfdom::urdfdom_sensor APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(urdfdom::urdfdom_sensor PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liburdfdom_sensor.so.3.0"
  IMPORTED_SONAME_RELEASE "liburdfdom_sensor.so.3.0"
  )

list(APPEND _cmake_import_check_targets urdfdom::urdfdom_sensor )
list(APPEND _cmake_import_check_files_for_urdfdom::urdfdom_sensor "${_IMPORT_PREFIX}/lib/liburdfdom_sensor.so.3.0" )

# Import target "urdfdom::urdfdom_model_state" for configuration "Release"
set_property(TARGET urdfdom::urdfdom_model_state APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(urdfdom::urdfdom_model_state PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liburdfdom_model_state.so.3.0"
  IMPORTED_SONAME_RELEASE "liburdfdom_model_state.so.3.0"
  )

list(APPEND _cmake_import_check_targets urdfdom::urdfdom_model_state )
list(APPEND _cmake_import_check_files_for_urdfdom::urdfdom_model_state "${_IMPORT_PREFIX}/lib/liburdfdom_model_state.so.3.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
