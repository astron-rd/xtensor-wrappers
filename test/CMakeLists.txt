# Correctness tests for the xtensor-wrappers library, compared against a raw
# FFTW reference. Plain runtime checks (no test framework) so they are active
# in Release builds where assert() is compiled out.
list(
  APPEND
  TESTS
  test_plan
  test_fft2
  test_fft2_rows_cols)

foreach(test ${TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE xtensor-wrappers)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
