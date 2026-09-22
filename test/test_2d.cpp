// 2D correctness of the xtensor-wrappers library.
// Verifies the native rank-2 transforms (the same plan API as 1D) against raw
// FFTW references.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include <xtensor-wrappers/plan.hpp>
#include <xtensor/containers/xarray.hpp>

#include "test_helpers.hpp"

template <class T>
static void check_2d_c2c(const std::vector<std::size_t> &shape) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c<T>(in);

  auto plan = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  REQUIRE(plan.output().shape() == in.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void check_2d_r2c(const std::vector<std::size_t> &shape) {
  auto in = random_real<T>(shape);
  auto ref = ref_r2c(in);

  auto plan = xt::fftw::make_rfft_plan(in, xt::xarray<std::complex<T>>{});
  REQUIRE(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void check_2d_irfft(const std::vector<std::size_t> &shape) {
  const bool odd = (shape.back() % 2 == 1);
  auto in = random_real<T>(shape);
  auto spec = ref_r2c(in); // half-complex, last dim n/2+1
  // Multidimensional c2r destroys its input by default (FFTW); reference the
  // transform on a copy so the original spectrum stays valid for the plan.
  auto spec_copy = spec;
  auto ref = ref_c2r<T>(spec_copy, odd);

  auto plan = xt::fftw::make_irfft_plan(
      spec, xt::xarray<T>{},
      {static_cast<int>(shape[0]), static_cast<int>(shape[1])}, FFTW_ESTIMATE);
  REQUIRE(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

TEST_CASE("2D c2c via the plan wrapper matches FFTW") {
  check_2d_c2c<float>({8, 8});
  check_2d_c2c<double>({6, 10});
}

TEST_CASE("2D r2c via the plan wrapper matches FFTW") {
  check_2d_r2c<float>({6, 8});  // even last dim
  check_2d_r2c<double>({5, 7}); // odd last dim
}

TEST_CASE("2D irfft via the plan wrapper matches FFTW") {
  check_2d_irfft<float>({4, 8});  // even inverse size
  check_2d_irfft<double>({5, 7}); // odd inverse size
}

// The caller-buffer overloads of make_*_plan() are rank-generic; a 2D input
// proves they cover the same ground as the 1D case with caller-owned buffers.
template <class T>
static void check_2d_c2c_external(const std::vector<std::size_t> &shape) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c<T>(in);

  std::vector<std::complex<T>> out(shape[0] * shape[1]);
  auto p = xt::fftw::make_fft_plan(
      in.data(), out.data(),
      {static_cast<int>(shape[0]), static_cast<int>(shape[1])});
  p.execute();

  xt::xarray<std::complex<T>> o(std::vector<std::size_t>{shape[0], shape[1]});
  std::copy_n(out.begin(), out.size(), o.begin());
  CHECK(allclose(o, ref));
}

template <class T>
static void check_2d_r2c_external(const std::vector<std::size_t> &shape) {
  auto in = random_real<T>(shape);
  auto ref = ref_r2c(in);
  const std::size_t half = shape.back() / 2 + 1;

  std::vector<std::complex<T>> out(shape[0] * half);
  auto p = xt::fftw::make_rfft_plan(
      in.data(), out.data(),
      {static_cast<int>(shape[0]), static_cast<int>(shape[1])});
  p.execute();

  xt::xarray<std::complex<T>> o(std::vector<std::size_t>{shape[0], half});
  std::copy_n(out.begin(), out.size(), o.begin());
  CHECK(allclose(o, ref));
}

template <class T>
static void check_2d_irfft_external(const std::vector<std::size_t> &shape) {
  const bool odd = (shape.back() % 2 == 1);
  auto in = random_real<T>(shape);
  auto spec = ref_r2c(in); // half-complex, last dim n/2+1
  auto spec_copy = spec;   // multidimensional c2r may destroy its input
  auto ref = ref_c2r<T>(spec_copy, odd);

  std::vector<T> out(shape[0] * shape[1]);
  auto p = xt::fftw::make_irfft_plan(
      spec.data(), out.data(),
      {static_cast<int>(shape[0]), static_cast<int>(shape[1])});
  p.execute();

  xt::xarray<T> o(std::vector<std::size_t>{shape[0], shape[1]});
  std::copy_n(out.begin(), out.size(), o.begin());
  CHECK(allclose(o, ref));
}

TEST_CASE("2D external buffers match the FFTW reference") {
  check_2d_c2c_external<double>({8, 8});
  check_2d_c2c_external<float>({6, 10});
  check_2d_r2c_external<double>({6, 8}); // even last dim
  check_2d_r2c_external<float>({5, 7});  // odd last dim
  check_2d_irfft_external<double>({4, 8});
  check_2d_irfft_external<float>({5, 7});
}
