cmake_minimum_required(VERSION 3.16)
project(tinyremo_projhess)

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
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
endif()

add_executable(projhess_test projhess_test.cpp)
target_include_directories(projhess_test PRIVATE ${doctest_SOURCE_DIR})
target_link_libraries(projhess_test PRIVATE Eigen3::Eigen)
target_compile_features(projhess_test PRIVATE cxx_std_20)
