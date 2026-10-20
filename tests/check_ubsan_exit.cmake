execute_process(COMMAND "${PROBE}" RESULT_VARIABLE status
  OUTPUT_VARIABLE output ERROR_VARIABLE diagnostic)
if(NOT diagnostic MATCHES "runtime error: signed integer overflow")
  message(FATAL_ERROR "Expected the deliberate UBSan diagnostic, got: ${output}${diagnostic}")
endif()
if("${status}" STREQUAL "0")
  message(FATAL_ERROR "UBSan reported signed overflow but exited successfully")
endif()
message(STATUS "UBSan reported signed overflow and failed with exit status ${status}")
