#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "octomath" for configuration "Release"
set_property(TARGET octomath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(octomath PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboctomath.so"
  IMPORTED_SONAME_RELEASE "liboctomath.so"
  )

list(APPEND _cmake_import_check_targets octomath )
list(APPEND _cmake_import_check_files_for_octomath "${_IMPORT_PREFIX}/lib/liboctomath.so" )

# Import target "octomath-static" for configuration "Release"
set_property(TARGET octomath-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(octomath-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboctomath.a"
  )

list(APPEND _cmake_import_check_targets octomath-static )
list(APPEND _cmake_import_check_files_for_octomath-static "${_IMPORT_PREFIX}/lib/liboctomath.a" )

# Import target "octomap" for configuration "Release"
set_property(TARGET octomap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(octomap PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboctomap.so"
  IMPORTED_SONAME_RELEASE "liboctomap.so"
  )

list(APPEND _cmake_import_check_targets octomap )
list(APPEND _cmake_import_check_files_for_octomap "${_IMPORT_PREFIX}/lib/liboctomap.so" )

# Import target "octomap-static" for configuration "Release"
set_property(TARGET octomap-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(octomap-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liboctomap.a"
  )

list(APPEND _cmake_import_check_targets octomap-static )
list(APPEND _cmake_import_check_files_for_octomap-static "${_IMPORT_PREFIX}/lib/liboctomap.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
