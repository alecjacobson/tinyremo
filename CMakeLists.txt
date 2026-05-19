cmake_minimum_required(VERSION 3.16)
project(tinyremo_tests)

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
  GIT_SHALLOW TRUE
)
# Populate without running doctest's CMakeLists (it requires CMake < 3.5 which
# CMake 4 no longer supports). We only need the single include directory.
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
endif()

add_executable(tests tests.cpp)
target_include_directories(tests PRIVATE ${doctest_SOURCE_DIR})
target_link_libraries(tests PRIVATE Eigen3::Eigen)
target_compile_features(tests PRIVATE cxx_std_20)
