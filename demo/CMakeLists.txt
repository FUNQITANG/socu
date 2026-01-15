# Build a tiny demo CUDA shared library from mykernel.cu.
#
# Output naming is platform-specific (decided by the active toolchain):
#   Windows => mykernel.dll (+ mykernel.lib import library)
#   Linux   => libmykernel.so
#   macOS   => libmykernel.dylib

find_package(CUDAToolkit REQUIRED)

add_library(mykernel SHARED mykernel.cu)

target_link_libraries(mykernel PRIVATE CUDA::cudart)

set_target_properties(mykernel PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Prefer shared CUDA runtime (cudart64_*.dll on Windows) to keep the DLL smaller.
set_property(TARGET mykernel PROPERTY CUDA_RUNTIME_LIBRARY Shared)

# Compiler flags
target_compile_options(mykernel PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Windows notes:
# - CUDA 13.1 officially supports VS 2019-2022.
#   Newer MSVC versions may require --allow-unsupported-compiler, which must be
#   applied before project(... CUDA). This is handled in the top-level CMakeLists.
# - /utf-8 avoids codepage warnings in some locales.
if(WIN32)
  target_compile_options(mykernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  )
endif()

# Optional: build an importable CPython extension module (.pyd on Windows).
# This still relies on the same CUDA code, but lets you `import socu._mykernel`
# and call functions without setting up ctypes signatures.
option(SOCU_BUILD_PYTHON_EXT "Build the CPython extension module socu._mykernel" OFF)

if(SOCU_BUILD_PYTHON_EXT)
  find_package(Python3 QUIET COMPONENTS Development.Module)
  if(NOT Python3_Development_Module_FOUND)
    message(WARNING "SOCU: Python3 Development.Module not found; skipping socu._mykernel extension build. Install a full CPython/conda with headers+libs, or provide Python3_INCLUDE_DIRS/Python3_LIBRARIES.")
  else()

  add_library(socu_mykernel_py MODULE mykernel_python_module.cpp)

  # Make the file importable as `socu._mykernel`.
  set_target_properties(socu_mykernel_py PROPERTIES
    OUTPUT_NAME "_mykernel"
    PREFIX ""
  )
  if(WIN32)
    set_target_properties(socu_mykernel_py PROPERTIES SUFFIX ".pyd")
  endif()

  target_link_libraries(socu_mykernel_py PRIVATE Python3::Module mykernel)

  if(WIN32)
    target_compile_options(socu_mykernel_py PRIVATE /utf-8)
  endif()

  # Install into the Python package folder for local development.
  set(SOCU_PYTHON_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/socu" CACHE PATH
      "Where to install Python extension modules for local import (e.g. socu).")
  if(SOCU_PYTHON_PACKAGE_DIR)
    install(TARGETS socu_mykernel_py
      RUNTIME DESTINATION "${SOCU_PYTHON_PACKAGE_DIR}"
      LIBRARY DESTINATION "${SOCU_PYTHON_PACKAGE_DIR}"
    )
  endif()
  endif()
endif()

# Install rules
#
# Goal: make the produced shared library easy to load from Python (ctypes/cffi)
# without hunting inside out/build/... directories.
#
# By default we install directly into the source tree under socu/_native.
# You can override this path at configure time via:
#   -DSOCU_PYTHON_NATIVE_DIR="..."
set(SOCU_PYTHON_NATIVE_DIR "${SOCU_PYTHON_NATIVE_DIR}" CACHE PATH
    "Where to install native libraries for Python loading (e.g. socu/_native).")

# Standard install layout (useful for packaging).
install(TARGETS mykernel
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Developer-friendly install: copy the shared library next to the Python package.
# Note: This uses an absolute DESTINATION (writes into the source tree by default).
if(SOCU_PYTHON_NATIVE_DIR)
  install(TARGETS mykernel
    RUNTIME DESTINATION "${SOCU_PYTHON_NATIVE_DIR}"
    LIBRARY DESTINATION "${SOCU_PYTHON_NATIVE_DIR}"
  )
endif()
