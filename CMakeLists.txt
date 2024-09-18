cmake_minimum_required(VERSION 3.18)  # Version depends on your CUDA version
project(GSAP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES ${ARCH})  # Set CUDA architectures

add_executable(run GSAP.cu)

target_compile_features(run PUBLIC cxx_std_17)

# Set CUDA flags
set_target_properties(run PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86"
)

# Define include directories
target_include_directories(run PRIVATE ${CMAKE_SOURCE_DIR}/GSAP)

# Specify compiler options for CUDA and C++
target_compile_options(run PRIVATE
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
  COMMAND ${CMAKE_COMMAND} -E remove run
)

# Enable testing
enable_testing()

# Add tests with different argument combinations
add_test(NAME Test_LL_1k COMMAND run LL 1k)
add_test(NAME Test_LH_1k COMMAND run LH 1k)
add_test(NAME Test_HL_1k COMMAND run HL 1k)
add_test(NAME Test_HH_1k COMMAND run HH 1k)

add_test(NAME Test_LL_5k COMMAND run LL 5k)
add_test(NAME Test_LH_5k COMMAND run LH 5k)
add_test(NAME Test_HL_5k COMMAND run HL 5k)
add_test(NAME Test_HH_5k COMMAND run HH 5k)

add_test(NAME Test_LL_20k COMMAND run LL 20k)
add_test(NAME Test_LH_20k COMMAND run LH 20k)
add_test(NAME Test_HL_20k COMMAND run HL 20k)
add_test(NAME Test_HH_20k COMMAND run HH 20k)

add_test(NAME Test_LL_50k COMMAND run LL 50k)
add_test(NAME Test_LH_50k COMMAND run LH 50k)
add_test(NAME Test_HL_50k COMMAND run HL 50k)
add_test(NAME Test_HH_50k COMMAND run HH 50k)
