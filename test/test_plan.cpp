// 1D correctness of the xtensor-fftw plan wrapper (float and double),
// checked against a raw FFTW reference plus an analytic reconstruction.

#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

template <class T>
static void test_1d_r2c(const std::size_t n) {
  auto in = random_real<T>({n});
  auto ref = ref_r2c(in);

  auto plan = xt::fftw::make_rfft_plan(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));

  // Reuse the same plan on fresh input (same pinned buffer), must match the
  // reference for the new data.
  in(0) += T(1);
  auto ref2 = ref_r2c(in);
  plan.execute();
  CHECK(allclose(plan.output(), ref2));
}

template <class T>
static void test_1d_c2c(const std::size_t n) {
  auto in = random_complex<T>({n});
  auto ref = ref_c2c<T>(in);

  auto plan = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == in.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void test_1d_irfft(const std::size_t n) {
  const bool odd = (n % 2 == 1);
  auto in = random_real<T>({n});
  auto spec = ref_r2c(in); // half-complex spectrum (n/2+1 entries)
  auto ref = ref_c2r<T>(spec, odd);

  auto plan =
      xt::fftw::make_irfft_plan(spec, xt::xarray<T>{}, odd, FFTW_ESTIMATE);
  CHECK(plan.output().shape() == std::vector<std::size_t>{n});
  plan.execute();
  CHECK(allclose(plan.output(), ref));

  // r2c followed by c2r reconstructs n*x (raw FFTW inverts without 1/n).
  for (std::size_t k = 0; k < n; ++k) {
    CHECK(near(plan.output()(k), in(k) * static_cast<T>(n)));
  }
}

int main() {
  test_1d_r2c<float>(16);
  test_1d_r2c<double>(100);
  test_1d_c2c<float>(128);
  test_1d_c2c<double>(63);
  test_1d_irfft<float>(8);     // even inverse size
  test_1d_irfft<float>(9);     // odd inverse size
  test_1d_irfft<double>(127);  // odd inverse size
  return report_and_exit("plan 1D (r2c/c2c/irfft, float and double)");
}
