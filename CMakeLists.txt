cmake_minimum_required(VERSION 3.18)  # Version depends on your CUDA version
project(MyCUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES ${ARCH})  # Set CUDA architectures

add_executable(GSAP main.cu)

target_compile_features(GSAP PUBLIC cxx_std_17)

# Set CUDA flags
set_target_properties(GSAP PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86"
)

# Define include directories
target_include_directories(GSAP PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Specify compiler options for CUDA and C++
target_compile_options(GSAP PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -use_fast_math
    -O3
    --extended-lambda
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    --extended-lambda
  >
)

# Add a custom target for cleaning up that doesn't use the reserved name "clean"
add_custom_target(cleanall
  COMMAND ${CMAKE_COMMAND} -E remove GSAP
)
