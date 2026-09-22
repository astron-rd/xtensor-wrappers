// Correctness of the 2D-FFT-by-row/column-decomposition wrapper
// (xt::fftw::plan_fft2 / fft2). The decomposed result must match both the raw
// raw FFTW 2D transform and the native rank-2 plan from the plan wrapper.

#include <stdexcept>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/fft2.hpp>
#include <xtensor-wrappers/plan.hpp>

#include "test_helpers.hpp"

template <class T>
static void test_decomposed(const std::vector<std::size_t> &shape) {
  auto in = random_complex<T>(shape);
  auto ref = ref_c2c<T>(in);

  // Native rank-2 plan as an independent check.
  auto native = xt::fftw::make_fft_plan(in, xt::xarray<std::complex<T>>{});
  native.execute();

  auto plan = xt::fftw::plan_fft2<T>(in, xt::xarray<std::complex<T>>{});
  CHECK(plan.output().shape() == ref.shape());
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

template <class T>
static void test_requires_2d() {
  auto in = random_complex<T>({8});
  bool threw = false;
  try {
    xt::fftw::plan_fft2<T> p(in, xt::xarray<std::complex<T>>{});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

int main() {
  test_decomposed<float>({8, 8});
  test_decomposed<float>({7, 5});
  test_decomposed<double>({6, 6});
  test_decomposed<float>({1, 8});  // single row
  test_decomposed<float>({8, 1});  // single column
  test_decomposed<float>({2, 2});
  test_requires_2d<float>();
  return report_and_exit("plan_fft2 row/column decomposition (float and double)");
}
