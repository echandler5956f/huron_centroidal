#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cmeel::tinyxml" for configuration "Release"
set_property(TARGET cmeel::tinyxml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cmeel::tinyxml PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtinyxml.so"
  IMPORTED_SONAME_RELEASE "libtinyxml.so"
  )

list(APPEND _cmake_import_check_targets cmeel::tinyxml )
list(APPEND _cmake_import_check_files_for_cmeel::tinyxml "${_IMPORT_PREFIX}/lib/libtinyxml.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
