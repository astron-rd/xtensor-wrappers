#ifndef XTENSOR_WRAPPERS_PLAN_1D_HPP
#define XTENSOR_WRAPPERS_PLAN_1D_HPP

#include <cassert>
#include <complex>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <xtensor-fftw/common.hpp>
#include <xtensor/containers/xarray.hpp>

#ifndef XTENSOR_FFTW_ASSERT
#define XTENSOR_FFTW_ASSERT(expr) assert(expr)
#endif

namespace xt::fftw {

namespace detail {

// Convert an xtensor shape to the int vector expected by the FFTW planning
// functions.
template <class S> inline std::vector<int> to_int_vector(const S &shape) {
  std::vector<int> result;
  result.reserve(shape.size());
  for (auto s : shape) {
    result.push_back(static_cast<int>(s));
  }
  return result;
}

// FFTW exposes the same API twice, under fftwf_* (single precision) and fftw_*
// (double precision). plan_traits<T> maps a scalar type to its FFTW functions
// so the plan machinery below is written exactly once.
template <class T> struct plan_traits;

template <> struct plan_traits<float> {
  using real_type = float;
  using complex_type = fftwf_complex;
  using plan_type = fftwf_plan;

  static plan_type make_r2c(int rank, const int *n, real_type *in,
                            complex_type *out, unsigned flags) {
    return fftwf_plan_dft_r2c(rank, n, in, out, flags);
  }

  static plan_type make_c2r(int rank, const int *n, complex_type *in,
                            real_type *out, unsigned flags) {
    return fftwf_plan_dft_c2r(rank, n, in, out, flags);
  }

  static plan_type make_c2c(int rank, const int *n, complex_type *in,
                            complex_type *out, int direction, unsigned flags) {
    return fftwf_plan_dft(rank, n, in, out, direction, flags);
  }

  static plan_type make_many_r2c(int rank, const int *n, int howmany,
                                 real_type *in, const int *inembed, int istride,
                                 int idist, complex_type *out,
                                 const int *onembed, int ostride, int odist,
                                 unsigned flags) {
    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                   idist, out, onembed, ostride, odist, flags);
  }

  static plan_type make_many_c2r(int rank, const int *n, int howmany,
                                 complex_type *in, const int *inembed,
                                 int istride, int idist, real_type *out,
                                 const int *onembed, int ostride, int odist,
                                 unsigned flags) {
    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                                   idist, out, onembed, ostride, odist, flags);
  }

  static plan_type make_many_c2c(int rank, const int *n, int howmany,
                                 complex_type *in, const int *inembed,
                                 int istride, int idist, complex_type *out,
                                 const int *onembed, int ostride, int odist,
                                 int direction, unsigned flags) {
    return fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                               out, onembed, ostride, odist, direction, flags);
  }

  static void execute(plan_type p) { fftwf_execute(p); }
  static void destroy_plan(plan_type p) { fftwf_destroy_plan(p); }
};

template <> struct plan_traits<double> {
  using real_type = double;
  using complex_type = fftw_complex;
  using plan_type = fftw_plan;

  static plan_type make_r2c(int rank, const int *n, real_type *in,
                            complex_type *out, unsigned flags) {
    return fftw_plan_dft_r2c(rank, n, in, out, flags);
  }

  static plan_type make_c2r(int rank, const int *n, complex_type *in,
                            real_type *out, unsigned flags) {
    return fftw_plan_dft_c2r(rank, n, in, out, flags);
  }

  static plan_type make_c2c(int rank, const int *n, complex_type *in,
                            complex_type *out, int direction, unsigned flags) {
    return fftw_plan_dft(rank, n, in, out, direction, flags);
  }

  static plan_type make_many_r2c(int rank, const int *n, int howmany,
                                 real_type *in, const int *inembed, int istride,
                                 int idist, complex_type *out,
                                 const int *onembed, int ostride, int odist,
                                 unsigned flags) {
    return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist,
                                  out, onembed, ostride, odist, flags);
  }

  static plan_type make_many_c2r(int rank, const int *n, int howmany,
                                 complex_type *in, const int *inembed,
                                 int istride, int idist, real_type *out,
                                 const int *onembed, int ostride, int odist,
                                 unsigned flags) {
    return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
                                  out, onembed, ostride, odist, flags);
  }

  static plan_type make_many_c2c(int rank, const int *n, int howmany,
                                 complex_type *in, const int *inembed,
                                 int istride, int idist, complex_type *out,
                                 const int *onembed, int ostride, int odist,
                                 int direction, unsigned flags) {
    return fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                              out, onembed, ostride, odist, direction, flags);
  }

  static void execute(plan_type p) { fftw_execute(p); }
  static void destroy_plan(plan_type p) { fftw_destroy_plan(p); }
};

} // namespace detail

/**
 * @brief RAII wrapper around an FFTW plan for either `float` (via
 *        `plan_float`) or `double` (via `plan_double`) precision.
 *
 * `StoredT` is the element type of the owned output buffer: `std::complex<T>`
 * for R2C/C2C plans, `T` for C2R plans.
 *
 * The transform's output buffer is owned by the plan (`output()`), which
 * removes the dangling/reallocation hazard on that side: it cannot be resized
 * or freed out from under the plan. The *input* buffer, however, is supplied
 * by the caller to `make_*_plan()` so that fresh data can be fed to
 * `execute()` repeatedly, and must therefore outlive the plan and must not be
 * resized or reallocated between plan creation and destruction.
 *
 * The class is move-only. FFTW plan creation and destruction are guarded by a
 * global mutex (FFTW planning is not thread-safe); `execute()` needs no lock
 * because FFTW >= 3.3.5 guarantees thread-safe execution.
 */
template <class T, class StoredT = std::complex<T>> class basic_plan {
public:
  using traits = detail::plan_traits<T>;
  using plan_type = typename traits::plan_type;
  using real_type = typename traits::real_type;
  using complex_type = typename traits::complex_type;
  using output_type = xt::xarray<StoredT>;

  basic_plan() = default;

  basic_plan(const basic_plan &) = delete;
  basic_plan &operator=(const basic_plan &) = delete;

  /**
   * @brief Move constructor.
   */
  basic_plan(basic_plan &&other) noexcept
      : m_plan(other.m_plan), m_output(std::move(other.m_output)),
        m_expected_output_size(other.m_expected_output_size) {
    other.m_plan = nullptr;
  }

  /**
   * @brief Move assignment operator.
   */
  basic_plan &operator=(basic_plan &&other) noexcept {
    if (this != &other) {
      std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
      if (m_plan) {
        traits::destroy_plan(m_plan);
      }
      m_plan = other.m_plan;
      m_output = std::move(other.m_output);
      m_expected_output_size = other.m_expected_output_size;
      other.m_plan = nullptr;
    }
    return *this;
  }

  /**
   * @brief Takes ownership of a raw FFTW plan and its output buffer.
   *
   * @throws std::runtime_error if `p` is null, i.e. the FFTW planner failed
   * (e.g. out of memory or invalid parameters). The check is at runtime rather
   * than via `XTENSOR_FFTW_ASSERT` because that macro compiles out in Release
   * builds, and a null handle would otherwise be passed to `fftw*_execute`.
   */
  explicit basic_plan(plan_type p, output_type output)
      : m_plan(p), m_output(std::move(output)),
        m_expected_output_size(m_output.size()) {
    if (p == nullptr) {
      throw std::runtime_error("XTENSOR-FFTW: FFTW plan creation failed");
    }
  }

  ~basic_plan() {
    if (m_plan) {
      std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
      traits::destroy_plan(m_plan);
    }
  }

  /**
   * @brief Executes the FFT plan.
   *
   * @warning The input array passed to `make_*_plan()` must remain valid and
   * its memory must not be relocated (no resizing/reallocation) until this
   * plan is destroyed.
   */
  void execute() const {
    XTENSOR_FFTW_ASSERT(
        m_plan != nullptr &&
        "execute() on a default-constructed or moved-from plan");
    XTENSOR_FFTW_ASSERT(m_output.size() == m_expected_output_size &&
                        "output buffer changed/released since plan creation");
    traits::execute(m_plan);
  }

  plan_type get() const noexcept { return m_plan; }

  const output_type &output() const noexcept { return m_output; }

  /**
   * @brief Detaches the output buffer from the plan. After this, `execute()`
   *        has no valid output buffer, so it must not be called again.
   */
  output_type release() { return std::move(m_output); }

private:
  plan_type m_plan = nullptr;
  output_type m_output;
  std::size_t m_expected_output_size = 0;
};

/**
 * @brief Creates a plan for a Real-to-Complex FFT.
 *
 * @param output passed by value on purpose: the plan takes ownership of its
 * own output buffer (copied/moved in here), so the caller's array is left
 * untouched unless it is moved from explicitly.
 */
template <class T>
inline basic_plan<T> make_rfft_plan(xt::xarray<T> &input,
                                    xt::xarray<std::complex<T>> output,
                                    unsigned flags = FFTW_ESTIMATE) {
  output.resize(output_shape_from_input(input, true, false));

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto shape = detail::to_int_vector(input.shape());
  auto p = detail::plan_traits<T>::make_r2c(
      static_cast<int>(shape.size()), shape.data(), input.data(),
      reinterpret_cast<typename detail::plan_traits<T>::complex_type *>(
          output.data()),
      flags);

  return basic_plan<T>(p, std::move(output));
}

/**
 * @brief Creates a plan for an Inverse Real-to-Complex FFT.
 */
template <class T>
inline basic_plan<T, T>
make_irfft_plan(xt::xarray<std::complex<T>> &input, xt::xarray<T> output,
                bool odd_last_dim = false, unsigned flags = FFTW_ESTIMATE) {
  output.resize(output_shape_from_input(input, false, true, odd_last_dim));

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto shape = detail::to_int_vector(output.shape());
  auto p = detail::plan_traits<T>::make_c2r(
      static_cast<int>(shape.size()), shape.data(),
      reinterpret_cast<typename detail::plan_traits<T>::complex_type *>(
          input.data()),
      output.data(), flags);

  return basic_plan<T, T>(p, std::move(output));
}

/**
 * @brief Creates a plan for a Complex-to-Complex FFT.
 */
template <class T>
inline basic_plan<T> make_fft_plan(xt::xarray<std::complex<T>> &input,
                                   xt::xarray<std::complex<T>> output,
                                   int direction = FFTW_FORWARD,
                                   unsigned flags = FFTW_ESTIMATE) {
  output.resize(input.shape());

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto shape = detail::to_int_vector(input.shape());
  auto p = detail::plan_traits<T>::make_c2c(
      static_cast<int>(shape.size()), shape.data(),
      reinterpret_cast<typename detail::plan_traits<T>::complex_type *>(
          input.data()),
      reinterpret_cast<typename detail::plan_traits<T>::complex_type *>(
          output.data()),
      direction, flags);

  return basic_plan<T>(p, std::move(output));
}

using plan_float = basic_plan<float>;
using plan_double = basic_plan<double>;

} // namespace xt::fftw

#endif // XTENSOR_WRAPPERS_PLAN_1D_HPP
