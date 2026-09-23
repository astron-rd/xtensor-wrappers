// Parallel 2-D (plan_fft2) correctness of the xtensor-wrappers library.
// Verifies the decomposed row-pass + strided-column-pass transform (native
// orientation, no transpose) against a per-plane raw FFTW 2-D reference, for
// single-plane and multi-plane inputs, both buffer models and both directions.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

namespace {

using namespace xt::fftw;

// Extract plane `p` (shape {R, C}) from a rank-2/3 array as an independent 2-D
// array the raw-FFTW reference helpers can act on.
template <class T>
xt::xarray<std::complex<T>> plane(const xt::xarray<std::complex<T>> &a,
                                  std::size_t p, std::size_t r, std::size_t c) {
  xt::xarray<std::complex<T>> out(std::vector<std::size_t>{r, c});
  std::copy_n(a.data() + p * r * c, r * c, out.begin());
  return out;
}

// Per-plane 2-D reference for a rank-2 ({r, c}) or rank-3 ({planes, r, c})
// array, at the given direction.
template <class T>
xt::xarray<std::complex<T>> ref_c2c_2d(const xt::xarray<std::complex<T>> &in,
                                       int direction = FFTW_FORWARD) {
  auto shape = in.shape();
  const std::size_t planes = shape.size() == 3 ? shape[0] : 1;
  const std::size_t r = shape[shape.size() - 2];
  const std::size_t c = shape.back();
  xt::xarray<std::complex<T>> out(in.shape());
  for (std::size_t p = 0; p < planes; ++p) {
    auto in2 = plane(in, p, r, c);
    auto ref = ref_c2c<T>(in2, direction);
    std::copy_n(ref.data(), r * c, out.data() + p * r * c);
  }
  return out;
}

template <class T>
static void check_fft2_owning(const std::vector<std::size_t> &shape,
                              int direction = FFTW_FORWARD) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c_2d(in, direction);

  auto plan = make_fft2_plan(in, direction);
  REQUIRE(plan.output().shape() == in.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
  plan.execute(); // re-executable on fresh data
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void check_fft2_external(const std::vector<int> &n,
                                int direction = FFTW_FORWARD) {
  std::vector<std::size_t> shape(n.begin(), n.end());
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c_2d(in, direction);

  std::vector<std::complex<T>> out(in.size());
  auto plan = make_fft2_plan(in.data(), out.data(), n, direction);
  plan.execute();

  xt::xarray<std::complex<T>> o(shape);
  std::copy_n(out.begin(), out.size(), o.begin());
  CHECK(allclose(o, ref));
}

} // namespace

TEST_CASE("parallel 2-D c2c (owning) matches the 2-D FFT reference") {
  check_fft2_owning<float>({16, 16});
  check_fft2_owning<double>({7, 11});  // non-square
  check_fft2_owning<double>({64, 32}); // non-square
}

TEST_CASE("parallel 2-D c2c (owning), multiple planes") {
  check_fft2_owning<float>({3, 32, 32});
  check_fft2_owning<double>({4, 7, 11}); // non-square per plane
}

TEST_CASE("parallel 2-D c2c supports the backward direction") {
  check_fft2_owning<double>({8, 8}, FFTW_BACKWARD);
  check_fft2_owning<float>({2, 9, 13}, FFTW_BACKWARD);
}

TEST_CASE("parallel 2-D c2c (caller buffers) matches the 2-D FFT reference") {
  check_fft2_external<double>({16, 16});
  check_fft2_external<float>({7, 11});
  check_fft2_external<double>({2, 32, 32});
  check_fft2_external<float>({3, 8, 12});
}

TEST_CASE("parallel 2-D plan output can be released") {
  auto in = random_complex<double>({8, 8});
  auto ref = ref_c2c_2d(in);
  auto plan = make_fft2_plan(in);
  plan.execute();
  auto out = plan.release();
  CHECK(allclose(out, ref));
  CHECK(plan.output().size() == 0); // detached
}

TEST_CASE("parallel 2-D plan rejects a non-2D/3D shape") {
  auto in = random_complex<double>({8});
  REQUIRE_THROWS_AS(make_fft2_plan(in), std::invalid_argument);

  std::vector<std::complex<double>> buf(8);
  REQUIRE_THROWS_AS(make_fft2_plan(buf.data(), buf.data(), std::vector<int>{8}),
                    std::invalid_argument);
}
