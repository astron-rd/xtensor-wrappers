# Correctness tests for the xtensor-wrappers library using the Catch2 test
# framework, compared against a raw FFTW reference.
list(
  APPEND
  TESTS
  test_plan
  test_fft2
  test_fft2_rows_cols)

foreach(test ${TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE xtensor-wrappers Catch2::Catch2WithMain)
  catch_discover_tests(${test})
endforeach()
