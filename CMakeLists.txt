cmake_minimum_required(VERSION 3.20)

# Top-level build entry.
#
# Notes on "cross-platform":
# - CMake generates build rules for the *current* platform + toolchain.
# - Ninja is a cross-platform build tool (works on Windows/Linux/macOS), but it
#   does not magically produce Linux/macOS binaries from a Windows toolchain.
#   The output format is decided by the active toolchain:
#     Windows => .dll (+ .lib import library)
#     Linux   => .so
#     macOS   => .dylib

# NVCC does a host-compiler compatibility check while CMake is identifying the
# CUDA compiler (this happens during project(... CUDA)). Target-level flags are
# evaluated *after* project(), so they are too late for compiler identification.
#
# CUDA 13.x officially supports VS 2019-2022; newer MSVC versions may need the
# override below.
if(WIN32)
  # NVCC picks a host compiler (cl.exe) for the host-side compilation steps.
  # For CUDA compiler identification, this must be configured *before* project(... CUDA).
  #
  # If you are using a newer Visual Studio version that CUDA does not support yet,
  # nvcc may crash (e.g. cudafe++ access violation) even when you pass
  # --allow-unsupported-compiler. In that case, point this at a supported VS 2022
  # toolset cl.exe (typically v143):
  #   -DSOCU_CUDA_HOST_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/<ver>/bin/Hostx64/x64/cl.exe"
  set(SOCU_CUDA_HOST_COMPILER "" CACHE FILEPATH "Path to NVCC host compiler (cl.exe). Leave empty to auto-detect from environment.")
  if(SOCU_CUDA_HOST_COMPILER)
    message(STATUS "SOCU: Using NVCC host compiler: ${SOCU_CUDA_HOST_COMPILER}")
    set(CMAKE_CUDA_HOST_COMPILER "${SOCU_CUDA_HOST_COMPILER}" CACHE FILEPATH "NVCC host compiler" FORCE)
  elseif(DEFINED ENV{SOCU_CUDA_HOST_COMPILER} AND NOT "$ENV{SOCU_CUDA_HOST_COMPILER}" STREQUAL "")
    message(STATUS "SOCU: Using NVCC host compiler from env SOCU_CUDA_HOST_COMPILER: $ENV{SOCU_CUDA_HOST_COMPILER}")
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{SOCU_CUDA_HOST_COMPILER}" CACHE FILEPATH "NVCC host compiler" FORCE)
  endif()

  option(SOCU_NVCC_ALLOW_UNSUPPORTED_COMPILER "Pass --allow-unsupported-compiler to nvcc (needed for newer MSVC versions)." ON)

  if(SOCU_NVCC_ALLOW_UNSUPPORTED_COMPILER)
    if(NOT "${CMAKE_CUDA_FLAGS_INIT}" MATCHES "(^| )--allow-unsupported-compiler( |$)")
      set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} --allow-unsupported-compiler")
    endif()

    # Also patch the cached value for existing build directories.
    if(DEFINED CMAKE_CUDA_FLAGS AND NOT "${CMAKE_CUDA_FLAGS}" MATCHES "(^| )--allow-unsupported-compiler( |$)")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler" CACHE STRING "CUDA flags" FORCE)
    endif()
  endif()
endif()

project(socu_cuda LANGUAGES CXX CUDA)

# Small quality-of-life option: allow disabling demo build (e.g., when packaging
# only the Python module).
option(SOCU_BUILD_DEMO "Build the CUDA demo shared library." ON)

# Put build outputs in predictable locations across generators.
# This is especially handy on Windows where the shared library is a .dll.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Multi-config generators (Visual Studio, Ninja Multi-Config) append the config
# name unless we provide per-config output dirs.
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${config}" config_uc)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_uc} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_uc} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_uc} "${CMAKE_BINARY_DIR}/lib")
endforeach()

# Default to a reasonable set; override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(SOCU_BUILD_DEMO)
  add_subdirectory(demo)
endif()
