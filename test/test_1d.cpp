// 1D correctness of the xtensor-fftw plan wrapper (float and double),
// checked against a raw FFTW reference plus an analytic reconstruction.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

template <class T> static void check_1d_r2c(const std::size_t n) {
  auto in = random_real<T>({n});
  auto ref = ref_r2c(in);

  auto plan = xt::fftw::make_rfft_plan(in, xt::xarray<std::complex<T>>{});
  REQUIRE(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));

  // Reuse the same plan on fresh input (same pinned buffer), must match the
  // reference for the new data.
  in(0) += T(1);
  auto ref2 = ref_r2c(in);
  plan.execute();
  CHECK(allclose(plan.output(), ref2));
}

template <class T> static void check_1d_c2c(const std::size_t n) {
  auto in = random_complex<T>({n});
  auto ref = ref_c2c<T>(in);

  auto plan = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == in.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T> static void check_1d_irfft(const std::size_t n) {
  const bool odd = (n % 2 == 1);
  auto in = random_real<T>({n});
  auto spec = ref_r2c(in); // half-complex spectrum (n/2+1 entries)
  auto ref = ref_c2r<T>(spec, odd);

  auto plan =
      xt::fftw::make_irfft_plan(spec, xt::xarray<T>{}, odd, FFTW_ESTIMATE);
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{n});
  plan.execute();
  CHECK(allclose(plan.output(), ref));

  // r2c followed by c2r reconstructs n*x (raw FFTW inverts without 1/n).
  for (std::size_t k = 0; k < n; ++k) {
    CHECK(near(plan.output()(k), in(k) * static_cast<T>(n)));
  }
}

TEST_CASE("1D r2c matches the FFTW reference") {
  check_1d_r2c<float>(16);
  check_1d_r2c<double>(100);
}

TEST_CASE("1D c2c matches the FFTW reference") {
  check_1d_c2c<float>(128);
  check_1d_c2c<double>(63);
}

TEST_CASE("1D irfft reconstructs the real signal") {
  check_1d_irfft<float>(8);    // even inverse size
  check_1d_irfft<float>(9);    // odd inverse size
  check_1d_irfft<double>(127); // odd inverse size
}

// The external-buffer ("into") variants write into caller-owned buffers.
namespace {
template <class T> static void check_1d_c2c_external(const std::size_t n) {
  auto in = random_complex<T>({n});
  auto ref = ref_c2c<T>(in);

  std::vector<std::complex<T>> out(n);
  auto p = xt::fftw::make_fft_plan_into(in.data(), out.data(),
                                        {static_cast<int>(n)});
  p.execute();

  xt::xarray<std::complex<T>> o(std::vector<std::size_t>{n});
  std::copy_n(out.begin(), n, o.begin());
  CHECK(allclose(o, ref));
}

template <class T> static void check_1d_r2c_external(const std::size_t n) {
  auto in = random_real<T>({n});
  auto ref = ref_r2c(in);
  const std::size_t half = n / 2 + 1;

  std::vector<std::complex<T>> out(half);
  auto p = xt::fftw::make_rfft_plan_into(in.data(), out.data(),
                                         {static_cast<int>(n)});
  p.execute();

  xt::xarray<std::complex<T>> o(std::vector<std::size_t>{half});
  std::copy_n(out.begin(), half, o.begin());
  CHECK(allclose(o, ref));
}

template <class T> static void check_1d_irfft_external(const std::size_t n) {
  const bool odd = (n % 2 == 1);
  const std::size_t half = n / 2 + 1;
  auto in = random_complex<T>({half});
  auto ref = ref_c2r<T>(in, odd);

  std::vector<T> out(n);
  auto p = xt::fftw::make_irfft_plan_into(in.data(), out.data(),
                                          {static_cast<int>(n)});
  p.execute();

  xt::xarray<T> o(std::vector<std::size_t>{n});
  std::copy_n(out.begin(), n, o.begin());
  CHECK(allclose(o, ref));
}

template <class T> static void check_1d_c2c_inplace(const std::size_t n) {
  auto in = random_complex<T>({n});
  auto ref = ref_c2c<T>(in);

  std::vector<std::complex<T>> buf(in.begin(), in.end());
  auto p = xt::fftw::make_fft_plan_into(buf.data(), buf.data(),
                                        {static_cast<int>(n)});
  p.execute();

  xt::xarray<std::complex<T>> o(std::vector<std::size_t>{n});
  std::copy_n(buf.begin(), n, o.begin());
  CHECK(allclose(o, ref));
}
} // namespace

TEST_CASE("1D external buffers match the FFTW reference") {
  check_1d_c2c_external<double>(1000);
  check_1d_r2c_external<float>(16);
  check_1d_r2c_external<double>(100);
  check_1d_irfft_external<double>(257);
}

TEST_CASE("1D c2c in-place over a caller buffer") {
  check_1d_c2c_inplace<float>(128);
  check_1d_c2c_inplace<double>(63);
}
