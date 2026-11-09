cmake_minimum_required(VERSION 3.18)
project(anndataoom_fast LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AOF_NATIVE "Tune for the build machine (-march=native); off for wheels" OFF)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
if(MSVC)
  # MSVC's default /openmp is OpenMP 2.0; the LLVM runtime has the 3.x API
  # (omp_set_max_active_levels) used here.
  add_library(aof_openmp INTERFACE)
  target_compile_options(aof_openmp INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
else()
  find_package(OpenMP REQUIRED)
  add_library(aof_openmp INTERFACE)
  target_link_libraries(aof_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()

# No -ffast-math and no implicit a*b+c -> fma contraction: results must be
# bit-identical across machines and between native and portable builds.
if(MSVC)
  set(AOF_FP_FLAGS /O2 /fp:precise /EHsc /bigobj)
else()
  set(AOF_FP_FLAGS -O3 -fno-math-errno -ffp-contract=off)
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# libdeflate (MIT), decompression side only, built into the extension.
set(LIBDEFLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libdeflate)
add_library(aof_deflate STATIC
  ${LIBDEFLATE_DIR}/lib/deflate_decompress.c
  ${LIBDEFLATE_DIR}/lib/zlib_decompress.c
  ${LIBDEFLATE_DIR}/lib/adler32.c
  ${LIBDEFLATE_DIR}/lib/utils.c
  ${LIBDEFLATE_DIR}/lib/x86/cpu_features.c
  ${LIBDEFLATE_DIR}/lib/arm/cpu_features.c)
target_include_directories(aof_deflate PUBLIC ${LIBDEFLATE_DIR})
set_target_properties(aof_deflate PROPERTIES C_VISIBILITY_PRESET hidden)

nanobind_add_module(_core NB_STATIC
  src/cpp/bindings.cpp
  src/cpp/reader.cpp
  src/cpp/kernels.cpp
  src/cpp/knn.cpp
  src/cpp/fuzzy.cpp
  src/cpp/umap.cpp
  src/cpp/groupstats.cpp)
target_link_libraries(_core PRIVATE aof_deflate aof_openmp)
target_compile_options(_core PRIVATE ${AOF_FP_FLAGS})
if(AOF_NATIVE AND NOT MSVC)
  target_compile_options(_core PRIVATE -march=native)
endif()

install(TARGETS _core LIBRARY DESTINATION anndataoom_fast)

# Exact port of leidenalg (RBConfigurationVertexPartition) -- separate module.
nanobind_add_module(_leiden_cpp NB_STATIC
  src/cpp/leiden/bindings_leiden.cpp
  src/cpp/leiden/leiden.cpp)
target_link_libraries(_leiden_cpp PRIVATE aof_openmp)
# Bit-exact floating point vs libleidenalg: no -ffast-math, no fma contraction.
target_compile_options(_leiden_cpp PRIVATE ${AOF_FP_FLAGS})
if(AOF_NATIVE AND NOT MSVC)
  target_compile_options(_leiden_cpp PRIVATE -march=native)
endif()
install(TARGETS _leiden_cpp LIBRARY DESTINATION anndataoom_fast)
