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
