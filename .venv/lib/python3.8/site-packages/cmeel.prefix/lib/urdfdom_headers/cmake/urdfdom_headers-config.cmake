if (urdfdom_headers_CONFIG_INCLUDED)
  return()
endif()
set(urdfdom_headers_CONFIG_INCLUDED TRUE)

set(urdfdom_headers_INCLUDE_DIRS "${urdfdom_headers_DIR}/../../../include/urdfdom_headers")

include("${urdfdom_headers_DIR}/urdfdom_headersExport.cmake")

list(APPEND urdfdom_headers_TARGETS urdfdom_headers::urdfdom_headers)
