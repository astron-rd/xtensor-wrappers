// 2D correctness of the xtensor-fftw plan wrapper: native rank-2 transforms
// (c2c, r2c, c2r) compared against a raw FFTW reference.

#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

template <class T>
static void test_2d_c2c(const std::vector<std::size_t> &shape) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c<T>(in);

  auto plan = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == in.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void test_2d_r2c(const std::vector<std::size_t> &shape) {
  auto in = random_real<T>(shape);
  auto ref = ref_r2c(in);

  auto plan = xt::fftw::make_rfft_plan(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

template <class T>
static void test_2d_irfft(const std::vector<std::size_t> &shape) {
  const bool odd = (shape.back() % 2 == 1);
  auto in = random_real<T>(shape);
  auto spec = ref_r2c(in); // half-complex, last dim n/2+1
  // Multidimensional c2r destroys its input by default (FFTW); reference the
  // transform on a copy so the original spectrum stays valid for the plan.
  auto spec_copy = spec;
  auto ref = ref_c2r<T>(spec_copy, odd);

  auto plan =
      xt::fftw::make_irfft_plan(spec, xt::xarray<T>{}, odd, FFTW_ESTIMATE);
  CHECK(plan.output().shape() == ref.shape());
  plan.execute();
  CHECK(allclose(plan.output(), ref));
}

int main() {
  test_2d_c2c<float>({8, 8});
  test_2d_c2c<double>({6, 10});
  test_2d_r2c<float>({6, 8});      // even last dim
  test_2d_r2c<double>({5, 7});     // odd last dim
  test_2d_irfft<float>({4, 8});    // even inverse size
  test_2d_irfft<double>({5, 7});   // odd inverse size
  return report_and_exit("plan 2D (c2c/r2c/irfft, float and double)");
}
