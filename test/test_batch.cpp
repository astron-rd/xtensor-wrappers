// Batched (guru plan_many) correctness of the xtensor-wrappers plan wrapper,
// checked against the raw FFTW reference in test_helpers. Exercises padded
// strides (e.g. a 1000-point FFT inside a 1024-element allocation), separate
// input/output layouts, element-level strides, and the owning vs caller-buffer
// factory overloads.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

// Distance between consecutive transforms on the input side (the borrowed
// output side is derived from the layout when needed).
std::size_t dist_of(const batch_layout &l) {
  const std::size_t per = per_of(l, true);
  const std::size_t dist = l.idist;
  return dist ? dist : per;
}

std::size_t input_vec_size(const batch_layout &l) {
  return (l.howmany - 1) * dist_of(l) + per_of(l, true);
}

template <class T>
std::vector<std::complex<T>> random_complex_batch(const batch_layout &l,
                                                  unsigned seed,
                                                  std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  std::vector<std::complex<T>> v(input_vec_size(l));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(T(-100), T(100));
  const std::size_t idist = dist_of(l);
  const std::size_t stride = static_cast<std::size_t>(l.istride);
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
  std::vector<T> v(input_vec_size(l));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(T(-100), T(100));
  const std::size_t idist = dist_of(l);
  const std::size_t stride = static_cast<std::size_t>(l.istride);
  for (std::size_t k = 0; k < l.howmany; ++k) {
    T *base = v.data() + k * idist;
    for (std::size_t i = 0; i < active_per; ++i) {
      base[i * stride] = dist(rng);
    }
  }
  return v;
}

// Owned output shape: {howmany} prepended to the per-transform output shape.
template <class T>
xt::xarray<std::complex<T>> slice_output(const std::complex<T> *out,
                                         std::size_t per, std::size_t stride,
                                         std::size_t k, std::size_t count) {
  const std::complex<T> *base = out + k * per;
  xt::xarray<std::complex<T>> x(std::vector<std::size_t>{count});
  for (std::size_t i = 0; i < count; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

template <class T>
xt::xarray<T> slice_output(const T *out, std::size_t per, std::size_t stride,
                           std::size_t k, std::size_t count) {
  const T *base = out + k * per;
  xt::xarray<T> x(std::vector<std::size_t>{count});
  for (std::size_t i = 0; i < count; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

template <class T>
xt::xarray<std::complex<T>> input_slice(const std::vector<std::complex<T>> &v,
                                        const batch_layout &l, std::size_t k,
                                        std::size_t active = 0) {
  const std::size_t active_per = active ? active : detail::product(l.n);
  const std::complex<T> *base = v.data() + k * dist_of(l);
  const std::size_t stride = static_cast<std::size_t>(l.istride);
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
  const T *base = v.data() + k * dist_of(l);
  const std::size_t stride = static_cast<std::size_t>(l.istride);
  xt::xarray<T> x(std::vector<std::size_t>{active_per});
  for (std::size_t i = 0; i < active_per; ++i) {
    x(i) = base[i * stride];
  }
  return x;
}

// Copy m*n contiguous row-major complex values into a 2D array; used for the
// per-transform slices of a rank-2 batch (tight layout, unit stride).
template <class T>
xt::xarray<std::complex<T>> to_2d(const std::complex<T> *base, std::size_t m,
                                  std::size_t n) {
  xt::xarray<std::complex<T>> x(std::vector<std::size_t>{m, n});
  std::copy_n(base, m * n, x.begin());
  return x;
}

template <class T>
xt::xarray<T> to_2d(const T *base, std::size_t m, std::size_t n) {
  xt::xarray<T> x(std::vector<std::size_t>{m, n});
  std::copy_n(base, m * n, x.begin());
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
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{k, n});
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto out = slice_output(plan.output().data(), n, 1, j, n);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched c2c: padded input, tight owned output") {
  using T = float;
  constexpr std::size_t k = 4, n = 64, in_alloc = 80;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {in_alloc};
  l.idist = in_alloc;

  auto in = random_complex_batch<T>(l, 7);
  auto plan = make_batch_fft_plan(in.data(), l);
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{k, n});
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto out = slice_output(plan.output().data(), n, 1, j, n);
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
    auto out = slice_output(plan.output().data(), n, 1, j, n);
    CHECK(allclose(out, ref));
  }
}

TEST_CASE("batched r2c matches the FFTW reference") {
  using T = double;
  constexpr std::size_t k = 6, n = 1000, alloc = 1024;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;

  auto in = random_real_batch<T>(l, 11);
  auto plan = make_batch_rfft_plan(in.data(), l);
  const std::size_t half = n / 2 + 1;
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{k, half});
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_r2c<T>(slice);
    auto out = slice_output(plan.output().data(), half, 1, j, half);
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
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{k, n});
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j, half);
    auto ref = ref_c2r<T>(slice, /*odd=*/true);
    auto out = slice_output(plan.output().data(), n, 1, j, n);
    CHECK(allclose(out, ref));
  }
}

// Tight layout: with idist/inembed left at their defaults the c2r factory must
// derive the input distance from the half-complex input size (n/2+1), not the
// real transform size n.
TEST_CASE("batched irfft: tight half-complex input") {
  using T = float;
  constexpr std::size_t k = 5, n = 257;
  const std::size_t half = n / 2 + 1;

  // Tightly packed half-complex rows: row j starts at j * half.
  std::vector<std::complex<T>> in(k * half);
  std::mt19937 rng(17);
  std::uniform_real_distribution<T> dist(T(-100), T(100));
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < half; ++i) {
      in[j * half + i] = {dist(rng), dist(rng)};
    }
  }

  batch_layout l;
  l.howmany = k;
  l.n = {static_cast<int>(n)};

  auto plan = make_batch_irfft_plan(in.data(), l);
  REQUIRE(plan.output().shape() == std::vector<std::size_t>{k, n});
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    xt::xarray<std::complex<T>> slice(std::vector<std::size_t>{half});
    for (std::size_t i = 0; i < half; ++i) {
      slice(i) = in[j * half + i];
    }
    auto ref = ref_c2r<T>(slice, /*odd=*/true);
    auto out = slice_output(plan.output().data(), n, 1, j, n);
    CHECK(allclose(out, ref));
  }
}

// Caller-buffer overloads: make_batch_*_plan(input, output, layout, ...) writes
// into the user's buffer; the output layout (onembed/ostride/odist) is honored.
TEST_CASE("batched c2c into caller buffers") {
  using T = double;
  constexpr std::size_t k = 8, n = 1000, alloc = 1024;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.inembed = {alloc};
  l.idist = alloc;

  auto in = random_complex_batch<T>(l, 42);
  std::vector<std::complex<T>> out(k * n);
  auto plan = make_batch_fft_plan(in.data(), out.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_c2c<T>(slice);
    auto got = slice_output(out.data(), n, 1, j, n);
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
  auto plan = make_batch_fft_plan(buf.data(), buf.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(original, l, j);
    auto ref = ref_c2c<T>(slice);
    auto got = slice_output(buf.data(), alloc, 1, j, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched r2c into caller buffers (padded output)") {
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
  std::vector<std::complex<T>> out(k * out_alloc);
  auto plan = make_batch_rfft_plan(in.data(), out.data(), l);
  plan.execute();
  const std::size_t half = n / 2 + 1;
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j);
    auto ref = ref_r2c<T>(slice);
    auto got = slice_output(out.data(), out_alloc, 1, j, half);
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
  std::vector<T> out(k * n);
  auto plan = make_batch_irfft_plan(in.data(), out.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = input_slice(in, l, j, half);
    auto ref = ref_c2r<T>(slice, /*odd=*/true);
    auto got = slice_output(out.data(), n, 1, j, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("owning batch plan rejects a padded output layout") {
  using T = float;
  constexpr std::size_t k = 2, n = 64;
  batch_layout l;
  l.howmany = k;
  l.n = {n};
  l.onembed = {n + 8};
  l.odist = n + 8; // padded output is only supported by the caller-buffer path

  std::vector<std::complex<T>> in(k * n);
  REQUIRE_THROWS_AS(make_batch_fft_plan(in.data(), l), std::invalid_argument);
  REQUIRE_THROWS_AS(
      make_batch_fft_plan(in.data(), l, FFTW_FORWARD, FFTW_ESTIMATE),
      std::invalid_argument);
}

// The batch factories derive rank from layout.n, so rank-2 (batched 2D) needs
// no extra code; these lock the 2D behaviour in for tight layouts.
TEST_CASE("batched 2D c2c (owning) matches the FFTW reference") {
  using T = double;
  constexpr std::size_t k = 4, m = 64, n = 48;
  batch_layout l;
  l.howmany = k;
  l.n = {static_cast<int>(m), static_cast<int>(n)};

  auto in = random_complex_batch<T>(l, 21);
  auto plan = make_batch_fft_plan(in.data(), l);
  REQUIRE(plan.output().shape() == (std::vector<std::size_t>{k, m, n}));
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = to_2d(in.data() + j * dist_of(l), m, n);
    auto ref = ref_c2c<T>(slice);
    auto got = to_2d(plan.output().data() + j * (m * n), m, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched 2D c2c into caller buffers") {
  using T = double;
  constexpr std::size_t k = 4, m = 64, n = 48;
  batch_layout l;
  l.howmany = k;
  l.n = {static_cast<int>(m), static_cast<int>(n)};

  auto in = random_complex_batch<T>(l, 21);
  std::vector<std::complex<T>> out(k * m * n);
  auto plan = make_batch_fft_plan(in.data(), out.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = to_2d(in.data() + j * dist_of(l), m, n);
    auto ref = ref_c2c<T>(slice);
    auto got = to_2d(out.data() + j * (m * n), m, n);
    CHECK(allclose(got, ref));
  }
}

TEST_CASE("batched 2D r2c into caller buffers") {
  using T = double;
  constexpr std::size_t k = 3, m = 32, n = 40;
  const std::size_t half = n / 2 + 1;
  batch_layout l;
  l.howmany = k;
  l.n = {static_cast<int>(m), static_cast<int>(n)};
  l.onembed = {static_cast<int>(m), static_cast<int>(half)};

  auto in = random_real_batch<T>(l, 31);
  std::vector<std::complex<T>> out(k * m * half);
  auto plan = make_batch_rfft_plan(in.data(), out.data(), l);
  plan.execute();
  for (std::size_t j = 0; j < k; ++j) {
    auto slice = to_2d(in.data() + j * dist_of(l), m, n);
    auto ref = ref_r2c<T>(slice);
    auto got = to_2d(out.data() + j * (m * half), m, half);
    CHECK(allclose(got, ref));
  }
}

// The batch is split into one chunk per OpenMP thread, but never so finely
// that a chunk falls below min_transforms_per_chunk transforms (doing so
// fragments the batch: a handful of tiny FFTW calls per chunk cannot exploit
// SIMD, which measurably regresses throughput on many-thread systems).
TEST_CASE("batch chunk count caps parallel splitting") {
  using namespace detail;
  // Capped by the transform count floor: at most howmany / 64 chunks.
  CHECK(batch_chunk_count(4096, 128) == 64);
  CHECK(batch_chunk_count(1024, 128) == 16);
  CHECK(batch_chunk_count(1024, 64) == 16);
  // Capped by the available threads when the batch is big enough.
  CHECK(batch_chunk_count(1024, 8) == 8);
  CHECK(batch_chunk_count(1024, 1) == 1);
  // Small batches stay single-chunk (serial, unchunked).
  CHECK(batch_chunk_count(64, 1024) == 1);
  CHECK(batch_chunk_count(8, 128) == 1);
  CHECK(batch_chunk_count(1, 128) == 1);

  // The resulting chunks are balanced and never smaller than the floor.
  for (std::size_t howmany : {128ul, 1024ul, 4096ul, 65536ul}) {
    std::vector<std::size_t> begins, counts;
    const std::size_t parts = batch_chunk_count(howmany, 128);
    chunk_ranges(howmany, parts, begins, counts);
    REQUIRE(!counts.empty());
    std::size_t total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      CHECK(counts[i] >= min_transforms_per_chunk);
      total += counts[i];
    }
    CHECK(total == howmany); // nothing dropped
  }
}
