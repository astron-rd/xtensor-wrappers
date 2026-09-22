// Batched (guru plan_many) correctness of the xtensor-wrappers plan wrapper,
// checked against the raw FFTW reference in test_helpers. Exercises padded
// strides (e.g. a 1000-point FFT inside a 1024-element allocation), separate
// input/output layouts, and element-level strides.

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

namespace {

using namespace xt::fftw;

std::size_t per_of(const batch_layout &l, bool input) {
  return detail::elements_per_transform(l, input ? l.inembed : l.onembed);
}

std::size_t dist_of(const batch_layout &l, bool input) {
  const std::size_t per = per_of(l, input);
  const std::size_t dist = input ? l.idist : l.odist;
  return dist ? dist : per;
}

std::size_t vec_size(const batch_layout &l, bool input) {
  return (l.howmany - 1) * dist_of(l, input) + per_of(l, input);
}

std::size_t stride_of(const batch_layout &l, bool input) {
  return static_cast<std::size_t>(input ? l.istride : l.ostride);
}

template <class T>
std::vector<std::complex<T>> random_complex_batch(const batch_layout &l,
                                                  unsigned seed,
                                                  std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  std::vector<std::complex<T>> v(vec_size(l, true));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(T(-100), T(100));
  const std::size_t idist = dist_of(l, true);
  const std::size_t stride = stride_of(l, true);
  for (std::size_t k = 0; k < l.howmany; ++k) {
    std::complex<T> *base = v.data() + k * idist;
    for (std::size_t i = 0; i < active_per; ++i) {
      base[i * stride] = {dist(rng), dist(rng)};
    }
  }
  return v;
}

template <class T>
std::vector<T> random_real_batch(const batch_layout &l, unsigned seed,
                                 std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  std::vector<T> v(vec_size(l, true));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(T(-100), T(100));
  const std::size_t idist = dist_of(l, true);
  const std::size_t stride = stride_of(l, true);
  for (std::size_t k = 0; k < l.howmany; ++k) {
    T *base = v.data() + k * idist;
    for (std::size_t i = 0; i < active_per; ++i) {
      base[i * stride] = dist(rng);
    }
  }
  return v;
}

template <class T>
xt::xarray<std::complex<T>> input_slice(const std::vector<std::complex<T>> &v,
                                        const batch_layout &l, std::size_t k,
                                        std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  const std::complex<T> *base = v.data() + k * dist_of(l, true);
  const std::size_t stride = stride_of(l, true);
  xt::xarray<std::complex<T>> x(std::vector<std::size_t>{active_per});
  for (std::size_t i = 0; i < active_per; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

template <class T>
xt::xarray<T> input_slice(const std::vector<T> &v, const batch_layout &l,
                          std::size_t k, std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  const T *base = v.data() + k * dist_of(l, true);
  const std::size_t stride = stride_of(l, true);
  xt::xarray<T> x(std::vector<std::size_t>{active_per});
  for (std::size_t i = 0; i < active_per; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

template <class T>
xt::xarray<std::complex<T>>
output_slice(const std::vector<std::complex<T>> &out, const batch_layout &l,
             std::size_t k, std::size_t count) {
  const std::complex<T> *base = out.data() + k * dist_of(l, false);
  const std::size_t stride = stride_of(l, false);
  xt::xarray<std::complex<T>> x(std::vector<std::size_t>{count});
  for (std::size_t i = 0; i < count; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

template <class T>
xt::xarray<T> output_slice(const std::vector<T> &out, const batch_layout &l,
                           std::size_t k, std::size_t count) {
  const T *base = out.data() + k * dist_of(l, false);
  const std::size_t stride = stride_of(l, false);
  xt::xarray<T> x(std::vector<std::size_t>{count});
  for (std::size_t i = 0; i < count; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

} // namespace

TEST_CASE("batched c2c: 1000-point FFT in 1024-element allocations") {
  using T = double;
  constexpr std::size_t k = 8, n = 1000, alloc = 1024;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;

  auto in = random_complex_batch<T>(l, 42);
  auto plan = make_batch_fft_plan(in.data(), l);
  REQUIRE(plan.output().size() == k * n); // tight output of n * howmany
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto out = output_slice(plan.output(), l, j, n);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched c2c: padded input and padded output") {
  using T = float;
  constexpr std::size_t k = 4, n = 64, in_alloc = 80, out_alloc = 96;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {in_alloc};
  l.idist = in_alloc;
  l.onembed = {out_alloc};
  l.odist = out_alloc;

  auto in = random_complex_batch<T>(l, 7);
  auto plan = make_batch_fft_plan(in.data(), l);
  REQUIRE(plan.output().size() == k * out_alloc);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto out = output_slice(plan.output(), l, j, n);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched c2c: element stride") {
  using T = double;
  constexpr std::size_t k = 3, n = 16, alloc = 32;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.istride = 2;
  l.idist = alloc;

  auto in = random_complex_batch<T>(l, 3);
  auto plan = make_batch_fft_plan(in.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto out = output_slice(plan.output(), l, j, n);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched r2c matches the FFTW reference") {
  using T = double;
  constexpr std::size_t k = 6, n = 1000, alloc = 1024, out_alloc = 640;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;
  l.onembed = {out_alloc};
  l.odist = out_alloc;

  auto in = random_real_batch<T>(l, 11);
  auto plan = make_batch_rfft_plan(in.data(), l);
  REQUIRE(plan.output().size() == k * out_alloc);
  plan.execute();
  const std::size_t half = n / 2 + 1;
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_r2c<T>(slice);
    auto out = output_slice(plan.output(), l, j, half);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched irfft reconstructs the real signal") {
  using T = double;
  constexpr std::size_t k = 5, n = 257, in_alloc = 160;
  const std::size_t half = n / 2 + 1;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {in_alloc};
  l.idist = in_alloc;

  auto in = random_complex_batch<T>(l, 13, half);
  auto plan = make_batch_irfft_plan(in.data(), l);
  REQUIRE(plan.output().size() == k * n); // tight real output
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j, half);
    auto ref = ref_c2r<T>(slice, /*odd=*/true);
    auto out = output_slice(plan.output(), l, j, n);
    CHECK(allclose(out, ref));
  }
}

// The external-buffer ("into") batch variants write into a caller-owned output
// buffer sized to the output footprint (vec_size(layout, false)).
TEST_CASE("batched c2c into caller buffers") {
  using T = double;
  constexpr std::size_t k = 8, n = 1000, alloc = 1024;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;

  auto in = random_complex_batch<T>(l, 42);
  std::vector<std::complex<T>> out(vec_size(l, false));
  auto plan = make_batch_fft_plan_into(in.data(), out.data(), l);
  REQUIRE(out.size() == k * n); // tight output of n * howmany
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto got = output_slice(out, l, j, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched c2c into caller buffers, in-place") {
  using T = float;
  constexpr std::size_t k = 5, n = 128, alloc = 160;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;
  l.onembed = {alloc}; // output footprint matches the input allocation
  l.odist = alloc;

  auto buf = random_complex_batch<T>(l, 5);
  auto original = buf;
  auto plan = make_batch_fft_plan_into(buf.data(), buf.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(original, l, j);
    auto ref = ref_c2c<T>(slice);
    auto got = output_slice(buf, l, j, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched r2c into caller buffers") {
  using T = double;
  constexpr std::size_t k = 6, n = 1000, alloc = 1024, out_alloc = 640;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;
  l.onembed = {out_alloc};
  l.odist = out_alloc;

  auto in = random_real_batch<T>(l, 11);
  std::vector<std::complex<T>> out(vec_size(l, false));
  auto plan = make_batch_rfft_plan_into(in.data(), out.data(), l);
  plan.execute();
  const std::size_t half = n / 2 + 1;
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_r2c<T>(slice);
    auto got = output_slice(out, l, j, half);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched irfft into caller buffers") {
  using T = double;
  constexpr std::size_t k = 5, n = 257, in_alloc = 160;
  const std::size_t half = n / 2 + 1;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {in_alloc};
  l.idist = in_alloc;

  auto in = random_complex_batch<T>(l, 13, half);
  std::vector<T> out(vec_size(l, false));
  auto plan = make_batch_irfft_plan_into(in.data(), out.data(), l);
  REQUIRE(out.size() == k * n); // tight real output
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j, half);
    auto ref = ref_c2r<T>(slice, /*odd=*/true);
    auto got = output_slice(out, l, j, n);
    CHECK(allclose(got, ref));
  }
}
