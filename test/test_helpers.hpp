// Shared helpers for the xtensor-wrappers tests: a CHECK macro that works in
// Release builds, approximate-equality comparisons, deterministic random data
// generation, and raw FFTW reference transforms used as ground truth.

#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

static int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

template <class T> constexpr T tolerance();

template <> constexpr float tolerance<float>() { return 1e-5f; }
template <> constexpr double tolerance<double>() { return 1e-12; }

// Approximate equality, relative to the magnitude of the values involved so it
// scales with the (unnormalized) FFT output.
template <class T>
static bool near(const std::complex<T> &a, const std::complex<T> &b, T tol) {
  return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

template <class T> static bool near(T a, T b, T tol) {
  return std::abs(a - b) <= tol * (T(1) + std::abs(a) + std::abs(b));
}

template <class T> static bool near(const std::complex<T> &a, const std::complex<T> &b) {
  return near<T>(a, b, tolerance<T>());
}

template <class T> static bool near(T a, T b) { return near<T>(a, b, tolerance<T>()); }

template <class T>
static bool allclose(const xt::xarray<T> &a, const xt::xarray<T> &b) {
  if (a.shape() != b.shape()) {
    return false;
  }
  return std::equal(a.begin(), a.end(), b.begin(),
                    [](T x, T y) { return near<T>(x, y); });
}

template <class T>
static bool allclose(const xt::xarray<std::complex<T>> &a,
                     const xt::xarray<std::complex<T>> &b) {
  if (a.shape() != b.shape()) {
    return false;
  }
  return std::equal(a.begin(), a.end(), b.begin(),
                    [](const std::complex<T> &x, const std::complex<T> &y) {
                      return near<T>(x, y);
                    });
}

// Deterministic pseudo-random data (LCG), scaled to [-100, 100]. The modulo
// must be cast to a signed type before subtracting, otherwise the subtraction
// wraps in unsigned arithmetic and yields huge garbage values.
template <class T> static T rand_scalar(unsigned &seed) {
  seed = seed * 1664525u + 1013904223u;
  const int value = static_cast<int>(seed % 200000u) - 100000;
  return static_cast<T>(value) / static_cast<T>(1000);
}

template <class T>
static xt::xarray<std::complex<T>> random_complex(const std::vector<std::size_t> &shape,
                                                  unsigned seed = 42) {
  xt::xarray<std::complex<T>> x(shape);
  for (auto &v : x) {
    v = std::complex<T>(rand_scalar<T>(seed), rand_scalar<T>(seed));
  }
  return x;
}

template <class T>
static xt::xarray<T> random_real(const std::vector<std::size_t> &shape,
                                 unsigned seed = 7) {
  xt::xarray<T> x(shape);
  for (auto &v : x) {
    v = rand_scalar<T>(seed);
  }
  return x;
}

// ---------------------------------------------------------------------------
// Raw FFTW references (ground truth), built directly on the FFTW planning
// functions via the library's plan_traits. Works for any rank.
// ---------------------------------------------------------------------------

template <class T>
static xt::xarray<std::complex<T>>
ref_c2c(xt::xarray<std::complex<T>> &input, int direction = FFTW_FORWARD) {
  using traits = xt::fftw::detail::plan_traits<T>;
  xt::xarray<std::complex<T>> out(input.shape());
  std::vector<int> n(input.shape().begin(), input.shape().end());
  std::lock_guard<std::mutex> guard(xt::fftw::detail::fftw_global_mutex());
  auto p = traits::make_c2c(
      static_cast<int>(n.size()), n.data(),
      reinterpret_cast<typename traits::complex_type *>(input.data()),
      reinterpret_cast<typename traits::complex_type *>(out.data()), direction,
      FFTW_ESTIMATE);
  traits::execute(p);
  traits::destroy_plan(p);
  return out;
}

template <class T>
static xt::xarray<std::complex<T>> ref_r2c(const xt::xarray<T> &input) {
  using traits = xt::fftw::detail::plan_traits<T>;
  xt::xarray<std::complex<T>> out(input.shape());
  auto shape = input.shape();
  shape.back() = shape.back() / 2 + 1;
  out.resize(shape);
  std::vector<int> n(input.shape().begin(), input.shape().end());
  std::lock_guard<std::mutex> guard(xt::fftw::detail::fftw_global_mutex());
  auto p = traits::make_r2c(
      static_cast<int>(n.size()), n.data(), const_cast<T *>(input.data()),
      reinterpret_cast<typename traits::complex_type *>(out.data()),
      FFTW_ESTIMATE);
  traits::execute(p);
  traits::destroy_plan(p);
  return out;
}

// Inverse transform; the real output length of the last dimension is derived
// from the half-complex input, matching output_shape_from_input().
template <class T>
static xt::xarray<T> ref_c2r(const xt::xarray<std::complex<T>> &input,
                             bool odd_last_dim = false) {
  using traits = xt::fftw::detail::plan_traits<T>;
  xt::xarray<T> out(input.shape());
  auto shape = input.shape();
  shape.back() =
      odd_last_dim ? 2 * (shape.back() - 1) + 1 : 2 * (shape.back() - 1);
  out.resize(shape);
  std::vector<int> n(shape.begin(), shape.end());
  std::lock_guard<std::mutex> guard(xt::fftw::detail::fftw_global_mutex());
  auto p = traits::make_c2r(
      static_cast<int>(n.size()), n.data(),
      reinterpret_cast<typename traits::complex_type *>(
          const_cast<std::complex<T> *>(input.data())),
      out.data(), FFTW_ESTIMATE);
  traits::execute(p);
  traits::destroy_plan(p);
  return out;
}

static int report_and_exit(const char *what) {
  if (g_failures == 0) {
    std::printf("PASS: %s\n", what);
    return 0;
  }
  std::printf("FAIL: %s: %d check(s) failed\n", what, g_failures);
  return 1;
}

#endif // TEST_HELPERS_HPP
