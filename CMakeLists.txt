cmake_minimum_required(VERSION 3.16)
project(comparison)

include(FetchContent)

# Fetch Eigen before TinyAD so TinyAD's "if (NOT TARGET Eigen3::Eigen)" guard
# picks up this version instead of whatever find_package finds on the system.
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  tinyad
  GIT_REPOSITORY https://github.com/patr-schm/tinyad.git
  GIT_TAG 75093e14ef0d7bb39657c5f3b2aba1251afaa38c
)
#FetchContent_MakeAvailable(tinyad)

FetchContent_GetProperties(tinyad)
if(NOT tinyad_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(tinyad)
  message(STATUS "tinyad_SOURCE_DIR: ${tinyad_SOURCE_DIR}")
  message(STATUS "tinyad_BINARY_DIR: ${tinyad_BINARY_DIR}")
  add_subdirectory(${tinyad_SOURCE_DIR} ${tinyad_BINARY_DIR})
endif()


function(add_comparison name)
  add_executable(${name} ${name}.cpp)
  target_compile_definitions(${name} PUBLIC SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${name} PUBLIC TinyAD)
  target_compile_features(${name} PRIVATE cxx_std_20)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${name} PRIVATE -ferror-limit=1)
  endif()
endfunction()

add_comparison(llt_comparison)
add_comparison(spring_comparison)
