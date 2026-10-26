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
  set(REFERENCE_MAIN src/platform/reference_main_windows.cpp)
  set(GENERATE_MAIN src/platform/generate_main_windows.cpp)
else()
  set(MAPPING_SOURCE src/platform/mapped_file_posix.cpp)
  set(INSPECT_MAIN src/platform/inspect_main_posix.cpp)
  set(REFERENCE_MAIN src/platform/reference_main_posix.cpp)
  set(GENERATE_MAIN src/platform/generate_main_posix.cpp)
endif()
set(NANOINFER_SOURCES ${MAPPING_SOURCE} src/model/loader.cpp src/ops/ref/ops.cpp
  src/runtime/reference.cpp)
add_library(nanoinfer STATIC ${NANOINFER_SOURCES})
target_sources(nanoinfer PRIVATE src/runtime/kv_cache.cpp src/ops/ref/cached_attention.cpp
  src/runtime/sampling.cpp src/runtime/cached.cpp src/runtime/fp_env.cpp src/kernels/gemm.cpp
  src/kernels/gemv.cpp src/kernels/attention.cpp src/runtime/thread_pool.cpp)
find_package(Threads REQUIRED)
target_link_libraries(nanoinfer PUBLIC Threads::Threads)
target_include_directories(nanoinfer PUBLIC include PRIVATE src)
target_link_libraries(nanoinfer PRIVATE project_options)

# Only the micro-kernel translation units get AVX2/FMA code generation. The
# rest of the library keeps the baseline ISA and selects them at run time.
if(MSVC)
  set(NANOINFER_AVX2_FLAGS /arch:AVX2)
else()
  set(NANOINFER_AVX2_FLAGS -mavx2 -mfma)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
  target_sources(nanoinfer PRIVATE src/kernels/gemm_avx2.cpp src/kernels/gemv_avx2.cpp)
  set_source_files_properties(src/kernels/gemm_avx2.cpp src/kernels/gemv_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "${NANOINFER_AVX2_FLAGS}")
  target_compile_definitions(nanoinfer PRIVATE NANOINFER_HAVE_AVX2_KERNEL)
endif()
if(NANOINFER_BENCHMARKS)
  add_executable(gemm_bench bench/gemm_bench.cpp bench/bench_main.cpp)
  add_executable(peak_bench bench/peak_bench.cpp bench/peak_avx2.cpp bench/bench_main.cpp)
  set_source_files_properties(bench/peak_avx2.cpp PROPERTIES COMPILE_OPTIONS
    "${NANOINFER_AVX2_FLAGS}")
  add_executable(gemv_bench bench/gemv_bench.cpp bench/bench_main.cpp)
  add_executable(stream_bench bench/stream_bench.cpp bench/peak_avx2.cpp bench/bench_main.cpp)
  add_executable(e2e_bench bench/e2e_bench.cpp)
  target_link_libraries(e2e_bench PRIVATE nanoinfer project_options)
  foreach(target IN ITEMS gemm_bench peak_bench gemv_bench stream_bench)
    target_link_libraries(${target} PRIVATE nanoinfer project_options benchmark::benchmark)
  endforeach()
endif()

add_executable(gpt2ppl src/tools/perplexity.cpp)
target_link_libraries(gpt2ppl PRIVATE nanoinfer project_options)
add_executable(gpt2inspect src/tools/inspect.cpp ${INSPECT_MAIN})
target_include_directories(gpt2inspect PRIVATE src)
target_link_libraries(gpt2inspect PRIVATE nanoinfer project_options)
add_executable(gpt2ref src/tools/reference.cpp ${REFERENCE_MAIN})
target_include_directories(gpt2ref PRIVATE src)
target_link_libraries(gpt2ref PRIVATE nanoinfer project_options)
add_executable(gpt2rt src/tools/generate.cpp ${GENERATE_MAIN})
target_include_directories(gpt2rt PRIVATE src)
target_link_libraries(gpt2rt PRIVATE nanoinfer project_options)
if(BUILD_TESTING)
  add_executable(nanoinfer_tests tests/test_main.cpp tests/test_mapped_file.cpp tests/test_loader.cpp
    tests/test_ref_ops.cpp tests/test_kv_cache.cpp tests/test_sampling.cpp tests/test_gemm.cpp
    tests/test_gemv.cpp tests/test_thread_pool.cpp tests/test_attention.cpp tests/test_quantized.cpp)
  target_link_libraries(nanoinfer_tests PRIVATE nanoinfer project_options GTest::gtest)
  add_test(NAME unit COMMAND nanoinfer_tests)
  set_tests_properties(unit PROPERTIES LABELS concurrency)
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
  add_executable(nanoinfer_cache_tests tests/test_main.cpp tests/test_cached_forward.cpp)
  target_link_libraries(nanoinfer_cache_tests PRIVATE nanoinfer project_options GTest::gtest)
  add_test(NAME cached_forward COMMAND nanoinfer_cache_tests)
  set_tests_properties(cached_forward PROPERTIES FIXTURES_REQUIRED tiny_model)
  add_test(NAME generation_cli COMMAND "${Python3_EXECUTABLE}" -m tests.generation_cli
    --executable "$<TARGET_FILE:gpt2rt>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden" --directory "${CMAKE_BINARY_DIR}/generation")
  set_tests_properties(generation_cli PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model)
  add_test(NAME cli_output COMMAND "${Python3_EXECUTABLE}" -m tests.cli_output
    --inspector "$<TARGET_FILE:gpt2inspect>" --generator "$<TARGET_FILE:gpt2rt>"
    --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin")
  set_tests_properties(cli_output PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model)
  add_executable(gpt2cachecheck tests/cache_check.cpp ${REFERENCE_MAIN})
  target_include_directories(gpt2cachecheck PRIVATE src)
  target_link_libraries(gpt2cachecheck PRIVATE nanoinfer project_options)
  add_test(NAME golden_cached COMMAND "${Python3_EXECUTABLE}" -m tests.golden_cached
    --executable "$<TARGET_FILE:gpt2cachecheck>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden" --directory "${CMAKE_BINARY_DIR}/cached-golden"
    --report "${CMAKE_BINARY_DIR}/cached-golden-report.json")
  set_tests_properties(golden_cached PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model)
  add_executable(gpt2opt tests/optimized_check.cpp ${REFERENCE_MAIN})
  target_include_directories(gpt2opt PRIVATE src)
  target_link_libraries(gpt2opt PRIVATE nanoinfer project_options)
  add_test(NAME golden_optimized COMMAND "${Python3_EXECUTABLE}" -m tests.golden_optimized
    --reference-executable "$<TARGET_FILE:gpt2ref>" --executable "$<TARGET_FILE:gpt2opt>"
    --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin" --golden "${CMAKE_BINARY_DIR}/fixtures/golden"
    --directory "${CMAKE_BINARY_DIR}/optimized-golden"
    --report "${CMAKE_BINARY_DIR}/optimized-golden-report.json"
    --kernels naive reordered blocked packed avx2)
  set_tests_properties(golden_optimized PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED "tiny_model;tiny_oracle")
  add_test(NAME thread_invariance COMMAND "${Python3_EXECUTABLE}" -m tests.thread_invariance
    --executable "$<TARGET_FILE:gpt2opt>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --directory "${CMAKE_BINARY_DIR}/thread-invariance")
  set_tests_properties(thread_invariance PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model LABELS concurrency)
  add_test(NAME oracle_tiny COMMAND "${Python3_EXECUTABLE}" -m python.oracle --tiny
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden")
  set_tests_properties(oracle_tiny PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model FIXTURES_SETUP tiny_oracle)
  add_test(NAME quantized_model COMMAND "${Python3_EXECUTABLE}" -m tests.quantized_model
    --inspector "$<TARGET_FILE:gpt2inspect>" --directory "${CMAKE_BINARY_DIR}/quantized")
  set_tests_properties(quantized_model PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME export_tiny_int8 COMMAND "${Python3_EXECUTABLE}" -m python.export --tiny
    --quant int8 --group 16 --output "${CMAKE_BINARY_DIR}/fixtures/tiny-int8.bin")
  set_tests_properties(export_tiny_int8 PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_SETUP tiny_int8)
  add_test(NAME oracle_tiny_int8 COMMAND "${Python3_EXECUTABLE}" -m python.oracle --tiny
    --quant int8 --group 16 --golden "${CMAKE_BINARY_DIR}/fixtures/golden")
  set_tests_properties(oracle_tiny_int8 PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model FIXTURES_SETUP tiny_int8_oracle)
  add_test(NAME golden_int8 COMMAND "${Python3_EXECUTABLE}" -m tests.golden_int8
    --executable "$<TARGET_FILE:gpt2opt>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny-int8.bin"
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden" --directory "${CMAKE_BINARY_DIR}/int8-golden"
    --report "${CMAKE_BINARY_DIR}/int8-golden-report.json")
  set_tests_properties(golden_int8 PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED "tiny_model;tiny_oracle;tiny_int8;tiny_int8_oracle" LABELS concurrency)
  add_test(NAME perplexity_consistency COMMAND "${Python3_EXECUTABLE}" -m
    tests.perplexity_consistency --executable "$<TARGET_FILE:gpt2ppl>"
    --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin" --directory "${CMAKE_BINARY_DIR}/perplexity")
  set_tests_properties(perplexity_consistency PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model LABELS concurrency)
  add_test(NAME demo_stream COMMAND "${Python3_EXECUTABLE}" -m tests.demo_stream
    --executable "$<TARGET_FILE:gpt2rt>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --directory "${CMAKE_BINARY_DIR}/demo-stream")
  set_tests_properties(demo_stream PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model LABELS concurrency)
  add_test(NAME loader_integration COMMAND "${Python3_EXECUTABLE}" -m tests.loader_integration
    --inspector "$<TARGET_FILE:gpt2inspect>" --directory "${CMAKE_BINARY_DIR}/awkward")
  set_tests_properties(loader_integration PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME model_schema COMMAND "${Python3_EXECUTABLE}" -m tests.model_schema
    --inspector "$<TARGET_FILE:gpt2inspect>" --directory "${CMAKE_BINARY_DIR}/schema")
  set_tests_properties(model_schema PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  add_test(NAME golden_tiny COMMAND "${Python3_EXECUTABLE}" -m tests.golden_forward
    --executable "$<TARGET_FILE:gpt2ref>" --model "${CMAKE_BINARY_DIR}/fixtures/tiny.bin"
    --golden "${CMAKE_BINARY_DIR}/fixtures/golden" --directory "${CMAKE_BINARY_DIR}/golden-check"
    --report "${CMAKE_BINARY_DIR}/golden-report.json")
  set_tests_properties(golden_tiny PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    FIXTURES_REQUIRED tiny_model)
  add_test(NAME reference_contract COMMAND "${Python3_EXECUTABLE}" -m tests.reference_contract
    --executable "$<TARGET_FILE:gpt2ref>" --directory "${CMAKE_BINARY_DIR}/reference-contract")
  set_tests_properties(reference_contract PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  # Mutation switches are private to test-only libraries. The normal library
  # and CLI never receive them, even when BUILD_TESTING is enabled.
  foreach(mutation IN ITEMS EXACT_GELU LAYERNORM_EPS)
    string(TOLOWER "${mutation}" suffix)
    add_library(nanoinfer_test_${suffix} STATIC ${NANOINFER_SOURCES})
    target_include_directories(nanoinfer_test_${suffix} PUBLIC include)
    target_link_libraries(nanoinfer_test_${suffix} PRIVATE project_options)
    target_compile_definitions(nanoinfer_test_${suffix} PRIVATE NANOINFER_TEST_${mutation})
    add_executable(gpt2ref_test_${suffix} src/tools/reference.cpp ${REFERENCE_MAIN})
    target_include_directories(gpt2ref_test_${suffix} PRIVATE src)
    target_link_libraries(gpt2ref_test_${suffix} PRIVATE
      nanoinfer_test_${suffix} project_options)
  endforeach()
  add_test(NAME golden_negative COMMAND "${Python3_EXECUTABLE}" -m tests.golden_negative
    --executable "$<TARGET_FILE:gpt2ref>"
    --erf-executable "$<TARGET_FILE:gpt2ref_test_exact_gelu>"
    --epsilon-executable "$<TARGET_FILE:gpt2ref_test_layernorm_eps>"
    --directory "${CMAKE_BINARY_DIR}/negative-golden"
    --report "${CMAKE_BINARY_DIR}/negative-report.json")
  set_tests_properties(golden_negative PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
