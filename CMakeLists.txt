cmake_minimum_required(VERSION 3.20)
project(nanoinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NANOINFER_ASAN "Enable address sanitizer (and UBSan on GCC/Clang)" OFF)
option(NANOINFER_TSAN "Enable thread sanitizer on Linux" OFF)
option(NANOINFER_NATIVE_RELEASE "Use host SIMD instructions in local Release benchmarks" OFF)
if(NANOINFER_ASAN AND NANOINFER_TSAN)
  message(FATAL_ERROR "ASan and TSan require separate builds")
endif()

add_library(project_options INTERFACE)
if(MSVC)
  target_compile_options(project_options INTERFACE /W4 /WX /permissive- /Zc:__cplusplus /utf-8)
  if(NANOINFER_NATIVE_RELEASE)
    target_compile_options(project_options INTERFACE "$<$<CONFIG:Release>:/O2;/arch:AVX2>")
  endif()
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
  if(NANOINFER_NATIVE_RELEASE)
    target_compile_options(project_options INTERFACE "$<$<CONFIG:Release>:-O3;-march=native>")
  endif()
  if(NANOINFER_ASAN)
    target_compile_options(project_options INTERFACE
      -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    target_link_options(project_options INTERFACE -fsanitize=address,undefined -fno-sanitize-recover=all)
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
  set(INSPECT_MAIN src/platform/inspect_main_windows.cpp)
else()
  set(MAPPING_SOURCE src/platform/mapped_file_posix.cpp)
  set(INSPECT_MAIN src/platform/inspect_main_posix.cpp)
endif()
add_library(nanoinfer STATIC ${MAPPING_SOURCE} src/model/loader.cpp)
target_include_directories(nanoinfer PUBLIC include)
target_link_libraries(nanoinfer PRIVATE project_options)
add_executable(gpt2inspect src/tools/inspect.cpp ${INSPECT_MAIN})
target_include_directories(gpt2inspect PRIVATE src)
target_link_libraries(gpt2inspect PRIVATE nanoinfer project_options)
if(BUILD_TESTING)
  add_executable(nanoinfer_tests tests/test_main.cpp tests/test_mapped_file.cpp tests/test_loader.cpp)
  target_link_libraries(nanoinfer_tests PRIVATE nanoinfer project_options GTest::gtest)
  add_test(NAME unit COMMAND nanoinfer_tests)
  if(NANOINFER_ASAN AND NOT MSVC)
    add_executable(ubsan_probe tests/ubsan_probe.cpp)
    target_link_libraries(ubsan_probe PRIVATE project_options)
    add_test(NAME ubsan_fails_on_error COMMAND "${CMAKE_COMMAND}"
      "-DPROBE=$<TARGET_FILE:ubsan_probe>" -P "${CMAKE_SOURCE_DIR}/tests/check_ubsan_exit.cmake")
  endif()
  if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv")
    # setup-python sets Python3_ROOT_DIR, which can override virtualenv discovery.
    find_program(NANOINFER_VENV_PYTHON NAMES python python3
      PATHS "${CMAKE_SOURCE_DIR}/.venv/Scripts" "${CMAKE_SOURCE_DIR}/.venv/bin"
      NO_DEFAULT_PATH REQUIRED)
    set(Python3_EXECUTABLE "${NANOINFER_VENV_PYTHON}")
  endif()
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  add_test(NAME export_tiny COMMAND "${Python3_EXECUTABLE}" -m python.export
    --tiny --output "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden")
  set_tests_properties(export_tiny PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_SETUP tiny_model)
  add_test(NAME roundtrip_tiny COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/roundtrip.py" --inspector "$<TARGET_FILE:gpt2inspect>"
    --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --manifest "${CMAKE_BINARY_DIR}/fixtures/tiny.json")
  set_tests_properties(roundtrip_tiny PROPERTIES FIXTURES_REQUIRED tiny_model)
  add_test(NAME loader_integration COMMAND "${Python3_EXECUTABLE}" -m tests.loader_integration
    --inspector "$<TARGET_FILE:gpt2inspect>" --directory "${CMAKE_BINARY_DIR}/awkward")
  set_tests_properties(loader_integration PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME model_schema COMMAND "${Python3_EXECUTABLE}" -m tests.model_schema
    --inspector "$<TARGET_FILE:gpt2inspect>" --directory "${CMAKE_BINARY_DIR}/schema")
  set_tests_properties(model_schema PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
