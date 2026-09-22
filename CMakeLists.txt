cmake_minimum_required(VERSION 3.14)
project(
  xtensor-wrappers
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Timings and tests only make sense in an optimised build; default to Release
# unless the user picked a configuration.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies.
#
# fftw3f (single precision) and fftw3 (double precision) are found via
# pkg-config under our own module names (XTW_*), so the resulting imported
# targets cannot clash with a consumer's FFTW/FFTWF targets.
#
# The xtensor stack is only fetched when this project is standalone. When it is
# consumed as a subproject (e.g. via FetchContent), the parent is expected to
# provide the `xtl` / `xtensor` / `xtensor-fftw` INTERFACE targets already.
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_search_module(XTW_FFTW REQUIRED fftw3f IMPORTED_TARGET)
pkg_search_module(XTW_FFTW_DOUBLE REQUIRED fftw3 IMPORTED_TARGET)

include(FetchContent)

if(NOT TARGET xtensor-fftw)
  # Standalone build: fetch the stack and give xtensor-fftw its
  # single-precision FFTW. FetchXTensor.cmake links against PkgConfig::FFTW.
  pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)
  set(XTENSOR_LIBRARIES xtensor xtensor-fftw)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchXTensor.cmake")
endif()

# ---------------------------------------------------------------------------
# xtensor-wrappers: header-only library of xtensor helpers built on top of the
# provided xtensor-fftw.
# ---------------------------------------------------------------------------
add_library(xtensor-wrappers INTERFACE)
# CMAKE_CURRENT_SOURCE_DIR, not CMAKE_SOURCE_DIR: the latter is the parent
# project's source dir when this library is consumed as a subproject.
target_include_directories(
  xtensor-wrappers INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
# xtensor is linked explicitly (not just via xtensor-fftw): the public API is
# built on xt::xarray, and a consumer's xtensor-fftw target may not carry the
# xtensor include path transitively.
target_link_libraries(xtensor-wrappers INTERFACE xtensor xtensor-fftw
                                                 PkgConfig::XTW_FFTW_DOUBLE)
target_compile_features(xtensor-wrappers INTERFACE cxx_std_20)

# Parallel batched plans. When enabled, batched plans split their transforms
# into contiguous chunks and execute them with #pragma omp parallel for over
# the batch items (completely transparent to the user). When disabled (or when
# OpenMP is unavailable) batches run single-threaded.
option(XTENSOR_WRAPPERS_USE_OPENMP "Run batched plans in parallel with OpenMP" ON)
if(XTENSOR_WRAPPERS_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_compile_definitions(xtensor-wrappers INTERFACE XTENSOR_WRAPPERS_USE_OPENMP=1)
  target_link_libraries(xtensor-wrappers INTERFACE OpenMP::OpenMP_CXX)
endif()

# Unit tests. Only the top-level project builds them (and only then does the
# Catch2 dependency get fetched); a consumer that pulls this library in as a
# subproject gets the library target and none of the test machinery.
set(XTENSOR_WRAPPERS_BUILD_TESTS_DEFAULT OFF)
if(PROJECT_IS_TOP_LEVEL)
  set(XTENSOR_WRAPPERS_BUILD_TESTS_DEFAULT ON)
endif()
option(XTENSOR_WRAPPERS_BUILD_TESTS "Build the unit tests"
       ${XTENSOR_WRAPPERS_BUILD_TESTS_DEFAULT})
if(XTENSOR_WRAPPERS_BUILD_TESTS)
  enable_testing()

  # Catch2 test framework (Catch2::Catch2WithMain for the test executables)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.16.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)

  add_subdirectory(test)
endif()
