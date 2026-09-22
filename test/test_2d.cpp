// 2D correctness of the xtensor-wrappers library.
// Verifies both native rank-2 transforms and the decomposed row/column
// implementation against raw FFTW references.

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
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

  auto plan =
      xt::fftw::make_irfft_plan(spec, xt::xarray<T>{}, odd, FFTW_ESTIMATE);
  REQUIRE(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void check_decomposed(const std::vector<std::size_t> &shape) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c<T>(in);

  // Native rank-2 plan as an independent check.
  auto native = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  native.execute();

  auto plan = xt::fftw::plan_fft2<T>(in, xt::xarray<std::complex<T>>{});
  REQUIRE(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
  CHECK(allclose(plan.output(), native.output()));

  // Reuse the decomposed plan on fresh input.
  in(0) += T(1);
  ref = ref_c2c<T>(in);
  plan.execute();
  CHECK(allclose(plan.output(), ref));

  // Convenience free function yields the same result.
  auto out = xt::fftw::fft2(in);
  CHECK(allclose(out, ref));

  // Moves keep the decomposed plan usable.
  auto moved = std::move(plan);
  in(0) += T(1);
  ref = ref_c2c<T>(in);
  moved.execute();
  CHECK(allclose(moved.output(), ref));
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

TEST_CASE("plan_fft2 row/column decomposition matches FFTW") {
  check_decomposed<float>({8, 8});
  check_decomposed<float>({7, 5});
  check_decomposed<double>({6, 6});
  check_decomposed<float>({1, 8}); // single row
  check_decomposed<float>({8, 1}); // single column
  check_decomposed<float>({2, 2});
}

TEST_CASE("plan_fft2 requires a 2-dimensional input") {
  auto in = random_complex<float>({8});
  REQUIRE_THROWS_AS(
      xt::fftw::plan_fft2<float>(in, xt::xarray<std::complex<float>>{}),
      std::invalid_argument);
}
