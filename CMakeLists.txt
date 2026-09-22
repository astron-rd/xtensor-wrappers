cmake_minimum_required(VERSION 3.14)
project(xtensor-wrappers VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Timings and tests only make sense in an optimised build; default to Release
# unless the user picked a configuration.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# Find FFTW (single and double precision).
#
# FetchXTensor.cmake links PkgConfig::FFTW (single precision) into the
# xtensor-fftw INTERFACE target, so fftw3f must be found before it is included.
# fftw3 (double precision) is needed by the tests that exercise the
# double-precision paths of the wrapper.
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3f IMPORTED_TARGET)
pkg_search_module(FFTW_DOUBLE REQUIRED fftw3 IMPORTED_TARGET)

# Fetch the xtensor stack (xtl, xtensor, xtensor-fftw)
set(XTENSOR_LIBRARIES xtensor xtensor-fftw)
include("${CMAKE_SOURCE_DIR}/cmake/FetchXTensor.cmake")

# ---------------------------------------------------------------------------
# xtensor-wrappers: header-only library of xtensor helpers built on top of the
# fetched xtensor-fftw.
# ---------------------------------------------------------------------------
add_library(xtensor-wrappers INTERFACE)
target_include_directories(xtensor-wrappers
                           INTERFACE "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")
target_link_libraries(xtensor-wrappers INTERFACE xtensor-fftw PkgConfig::FFTW_DOUBLE)
target_compile_features(xtensor-wrappers INTERFACE cxx_std_20)
