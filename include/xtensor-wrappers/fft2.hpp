#ifndef XTENSOR_WRAPPERS_FFT2_HPP
#define XTENSOR_WRAPPERS_FFT2_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan.hpp>

namespace xt { namespace fftw {

namespace detail {

// In-place 1D complex-to-complex DFT, pinned to a fixed contiguous buffer.
//
// Unlike xt::fftw::basic_plan this binds a raw pointer instead of owning an
// xarray: it lets a 2D plan hold one small plan per line length and re-execute
// it on freshly copied data, so no per-row plans are needed. Move-only and
// RAII; creation/destruction are guarded by the global FFTW mutex.
template <class T>
class fft1_c2c {
public:
  using plan_type = typename detail::plan_traits<T>::plan_type;

  fft1_c2c() = default;

  fft1_c2c(const fft1_c2c &) = delete;
  fft1_c2c &operator=(const fft1_c2c &) = delete;

  fft1_c2c(fft1_c2c &&other) noexcept : m_plan(other.m_plan), m_n(other.m_n) {
    other.m_plan = nullptr;
  }

  fft1_c2c &operator=(fft1_c2c &&other) noexcept {
    if (this != &other) {
      reset();
      m_plan = other.m_plan;
      m_n = other.m_n;
      other.m_plan = nullptr;
    }
    return *this;
  }

  ~fft1_c2c() { reset(); }

  explicit fft1_c2c(std::complex<T> *buffer, std::size_t n,
                    int direction = FFTW_FORWARD,
                    unsigned flags = FFTW_ESTIMATE)
      : m_n(n) {
    const int N = static_cast<int>(n);
    auto *c = reinterpret_cast<typename detail::plan_traits<T>::complex_type *>(
        buffer);
    std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
    m_plan = detail::plan_traits<T>::make_c2c(1, &N, c, c, direction, flags);
    if (m_plan == nullptr) {
      throw std::runtime_error(
          "XTENSOR-WRAPPERS: FFTW plan creation failed");
    }
  }

  void execute() const {
    if (m_plan == nullptr) {
      throw std::runtime_error("XTENSOR-WRAPPERS: execute() on an empty plan");
    }
    detail::plan_traits<T>::execute(m_plan);
  }

  std::size_t size() const noexcept { return m_n; }

private:
  void reset() noexcept {
    if (m_plan) {
      std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
      detail::plan_traits<T>::destroy_plan(m_plan);
      m_plan = nullptr;
    }
  }

  plan_type m_plan = nullptr;
  std::size_t m_n = 0;
};

} // namespace detail

/**
 * @brief Reusable 2D complex-to-complex FFT that decomposes the transform into
 *        1D FFTs over rows followed by 1D FFTs over columns.
 *
 * A multidimensional DFT is separable: transforming every row and then every
 * column with 1D FFTs equals the native n-dimensional transform (up to
 * rounding). The row and column passes reuse two small 1D plans that are
 * executed once per line, with a transpose in between so the column pass runs
 * over contiguous memory.
 *
 * As with `xt::fftw::make_fft_plan`, the input is supplied by the caller: it
 * must outlive the plan and must not be resized or reallocated while the plan
 * is alive. The output buffer is owned by the plan (`output()`/`release()`).
 *
 * @tparam T floating precision (`float` or `double`).
 */
template <class T>
class plan_fft2 {
public:
  using complex_type = std::complex<T>;
  using output_type = xt::xarray<complex_type>;

  plan_fft2(const plan_fft2 &) = delete;
  plan_fft2 &operator=(const plan_fft2 &) = delete;

  plan_fft2(plan_fft2 &&) noexcept = default;
  plan_fft2 &operator=(plan_fft2 &&) noexcept = default;

  /**
   * @brief Creates a 2D FFT plan for an m-by-n input.
   *
   * @param output moved into the plan and resized to the input shape.
   * @param direction FFTW direction (FFTW_FORWARD/FFTW_BACKWARD), applied to
   *        both axes.
   * @param flags FFTW planning flags (default FFTW_ESTIMATE).
   * @throws std::invalid_argument if the input is not 2-dimensional.
   */
  plan_fft2(xt::xarray<complex_type> &input,
            xt::xarray<complex_type> output,
            int direction = FFTW_FORWARD,
            unsigned flags = FFTW_ESTIMATE)
      : m_rows(size0(input)), m_cols(size1(input)), m_input(&input),
        m_output(std::move(output)), m_work(m_rows * m_cols),
        m_work_t(m_rows * m_cols), m_row_scratch(m_cols),
        m_col_scratch(m_rows),
        m_row_plan(m_row_scratch.data(), m_cols, direction, flags),
        m_col_plan(m_col_scratch.data(), m_rows, direction, flags) {
    m_output.resize(input.shape());
  }

  /**
   * @brief Runs the row/column decomposed 2D transform into the owned output.
   */
  void execute() {
    const std::size_t rows = m_rows;
    const std::size_t cols = m_cols;

    // Load the input into the working buffer.
    std::copy_n(m_input->data(), rows * cols, m_work.data());

    // Row pass: transform each contiguous row via the row scratch buffer.
    for (std::size_t r = 0; r < rows; ++r) {
      complex_type *row = m_work.data() + r * cols;
      std::copy_n(row, cols, m_row_scratch.data());
      m_row_plan.execute();
      std::copy_n(m_row_scratch.data(), cols, row);
    }

    // Transpose so each old column becomes a contiguous row of m_work_t.
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) {
        m_work_t.data()[c * rows + r] = m_work.data()[r * cols + c];
      }
    }

    // Column pass on the transposed data.
    for (std::size_t c = 0; c < cols; ++c) {
      complex_type *column = m_work_t.data() + c * rows;
      std::copy_n(column, rows, m_col_scratch.data());
      m_col_plan.execute();
      std::copy_n(m_col_scratch.data(), rows, column);
    }

    // Transpose back into the output.
    for (std::size_t c = 0; c < cols; ++c) {
      for (std::size_t r = 0; r < rows; ++r) {
        m_output.data()[r * cols + c] = m_work_t.data()[c * rows + r];
      }
    }
  }

  const output_type &output() const noexcept { return m_output; }

  /**
   * @brief Detaches the output buffer from the plan.
   */
  output_type release() { return std::move(m_output); }

private:
  static std::size_t size0(const xt::xarray<complex_type> &in) {
    require_2d(in);
    return in.shape()[0];
  }

  static std::size_t size1(const xt::xarray<complex_type> &in) {
    require_2d(in);
    return in.shape()[1];
  }

  static void require_2d(const xt::xarray<complex_type> &in) {
    if (in.dimension() != 2) {
      throw std::invalid_argument(
          "XTENSOR-WRAPPERS: plan_fft2 requires a 2-dimensional input");
    }
  }

  std::size_t m_rows;
  std::size_t m_cols;
  xt::xarray<complex_type> *m_input;
  output_type m_output;
  std::vector<complex_type> m_work;
  std::vector<complex_type> m_work_t;
  std::vector<complex_type> m_row_scratch;
  std::vector<complex_type> m_col_scratch;
  detail::fft1_c2c<T> m_row_plan;
  detail::fft1_c2c<T> m_col_plan;
};

/**
 * @brief Convenience wrapper: transforms a 2D array and returns the result.
 */
template <class T>
inline xt::xarray<std::complex<T>>
fft2(xt::xarray<std::complex<T>> &input, int direction = FFTW_FORWARD,
     unsigned flags = FFTW_ESTIMATE) {
  plan_fft2<T> p(input, xt::xarray<std::complex<T>>{}, direction, flags);
  p.execute();
  return p.release();
}

} // namespace fftw
} // namespace xt

#endif // XTENSOR_WRAPPERS_FFT2_HPP
