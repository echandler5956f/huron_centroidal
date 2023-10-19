#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pinocchio::pinocchio_double" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_double APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_double PROPERTIES
  IMPORTED_LOCATION_RELEASE "${PACKAGE_PREFIX_DIR}/lib/libpinocchio_double.so.2.9.2"
  IMPORTED_SONAME_RELEASE "libpinocchio_double.so.2.9.2"
  )

list(APPEND _cmake_import_check_targets pinocchio::pinocchio_double )
list(APPEND _cmake_import_check_files_for_pinocchio::pinocchio_double "${PACKAGE_PREFIX_DIR}/lib/libpinocchio_double.so.2.9.2" )

# Import target "pinocchio::pinocchio_pywrap_default" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_pywrap_default APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_pywrap_default PROPERTIES
  IMPORTED_LOCATION_RELEASE "${PACKAGE_PREFIX_DIR}/lib/python3.8/site-packages/pinocchio/pinocchio_pywrap_default.cpython-38-x86_64-linux-gnu.so.2.9.2"
  IMPORTED_SONAME_RELEASE "pinocchio_pywrap_default.cpython-38-x86_64-linux-gnu.so.2.9.2"
  )

list(APPEND _cmake_import_check_targets pinocchio::pinocchio_pywrap_default )
list(APPEND _cmake_import_check_files_for_pinocchio::pinocchio_pywrap_default "${PACKAGE_PREFIX_DIR}/lib/python3.8/site-packages/pinocchio/pinocchio_pywrap_default.cpython-38-x86_64-linux-gnu.so.2.9.2" )

# Import target "pinocchio::pinocchio_pywrap_casadi" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_pywrap_casadi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_pywrap_casadi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${PACKAGE_PREFIX_DIR}/lib/python3.8/site-packages/pinocchio/pinocchio_pywrap_casadi.cpython-38-x86_64-linux-gnu.so.2.9.2"
  IMPORTED_SONAME_RELEASE "pinocchio_pywrap_casadi.cpython-38-x86_64-linux-gnu.so.2.9.2"
  )

list(APPEND _cmake_import_check_targets pinocchio::pinocchio_pywrap_casadi )
list(APPEND _cmake_import_check_files_for_pinocchio::pinocchio_pywrap_casadi "${PACKAGE_PREFIX_DIR}/lib/python3.8/site-packages/pinocchio/pinocchio_pywrap_casadi.cpython-38-x86_64-linux-gnu.so.2.9.2" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
