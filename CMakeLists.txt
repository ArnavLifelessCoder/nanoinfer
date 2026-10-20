cmake_minimum_required(VERSION 3.20)
project(nanoinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NANOINFER_ASAN "Enable address sanitizer (and UBSan on GCC/Clang)" OFF)
option(NANOINFER_TSAN "Enable thread sanitizer on Linux" OFF)
if(NANOINFER_ASAN AND NANOINFER_TSAN)
  message(FATAL_ERROR "ASan and TSan require separate builds")
endif()

add_library(project_options INTERFACE)
if(MSVC)
  target_compile_options(project_options INTERFACE /W4 /WX /permissive- /Zc:__cplusplus /utf-8)
  if(NANOINFER_ASAN)
    # /RTC and incremental linking are incompatible with MSVC AddressSanitizer.
    # Instrument dependencies too: MSVC STL annotations must agree across objects.
    set(CMAKE_CXX_FLAGS_DEBUG "/O1 /Zi /fsanitize=address" CACHE STRING "" FORCE)
    target_compile_options(project_options INTERFACE /fsanitize=address /Zi)
    target_link_options(project_options INTERFACE /INCREMENTAL:NO /DEBUG)
  endif()
  if(NANOINFER_TSAN)
    message(FATAL_ERROR "TSan is supported only by the Linux preset")
  endif()
else()
  target_compile_options(project_options INTERFACE -Wall -Wextra -Wpedantic -Werror)
  if(NANOINFER_ASAN)
    target_compile_options(project_options INTERFACE
      -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(project_options INTERFACE -fsanitize=address,undefined)
  elseif(NANOINFER_TSAN)
    target_compile_options(project_options INTERFACE
      -O1 -g -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(project_options INTERFACE -fsanitize=thread)
  endif()
endif()

include(CTest)
include(FetchContent)
if(BUILD_TESTING)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

# Benchmarks are opt-in and run locally, never in CI.
option(NANOINFER_BENCHMARKS "Build local benchmarks" OFF)
if(NANOINFER_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

if(WIN32)
  set(MAPPING_SOURCE src/platform/mapped_file_windows.cpp)
else()
  set(MAPPING_SOURCE src/platform/mapped_file_posix.cpp)
endif()
add_library(nanoinfer STATIC ${MAPPING_SOURCE} src/model/loader.cpp)
target_include_directories(nanoinfer PUBLIC include)
target_link_libraries(nanoinfer PRIVATE project_options)
if(BUILD_TESTING)
  add_executable(nanoinfer_tests tests/test_main.cpp tests/test_mapped_file.cpp tests/test_loader.cpp)
  target_link_libraries(nanoinfer_tests PRIVATE nanoinfer project_options GTest::gtest)
  add_test(NAME unit COMMAND nanoinfer_tests)
endif()
