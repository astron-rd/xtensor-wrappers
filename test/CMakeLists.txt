# Correctness tests for the xtensor-wrappers library using the Catch2 test
# framework, compared against a raw FFTW reference.
list(APPEND TESTS test_1d test_2d test_batch test_fft2)

foreach(test ${TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE xtensor-wrappers Catch2::Catch2WithMain)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
