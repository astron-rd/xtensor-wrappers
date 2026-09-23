#ifndef XTENSOR_WRAPPERS_PLAN_2D_HPP
#define XTENSOR_WRAPPERS_PLAN_2D_HPP

// Parallel 2-D complex transform for the "few large transforms" workload (e.g.
// one 1024x1024 grid FFT per polarization) that the plain rank-N plan cannot
// parallelize: `basic_plan` runs FFTW single-threaded and `batch_plan`'s
// parallelism is bounded by `howmany`.
//
// A 2-D DFT is separable, so it is evaluated as two strided batched 1-D
// passes, each running through the library's OpenMP-over-batch machinery:
//
//   row pass   : batch n={cols}, howmany=rows*planes, istride=1,   idist=cols
//                (every row of every plane is one contiguous transform)
//   column pass: batch n={rows}, howmany=cols,       istride=cols, idist=1,
//                ostride=cols, odist=1 (every column is one strided transform)
//
// No intermediate transpose is needed: the column pass simply reads the rows
// with a strided access pattern and writes the result out with a matching
// strided output layout, so the output is the transform in the NATIVE
// orientation (element [plane][row][col] / [row][col] is the usual 2-D DFT).
// Both passes are unnormalized and honour `direction`/`flags` unchanged.
//
// Two buffer models mirror the single-transform API:
//   - plan_fft2<T> (make_fft2_plan(xarray)): owns a workspace and the output
//     xarray; input is borrowed.
//   - external_fft2_plan<T> (make_fft2_plan(raw pointers)): borrows both input
//     and output buffers; the workspace is owned internally.

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan_batch.hpp>

namespace xt::fftw {

namespace detail {

// The plan handles for one decomposed 2-D transform: an external row plan over
// caller buffers and one strided column plan per plane. `work` is the
// intermediate workspace between the two passes (owned by the caller of the
// builder), laid out natively as {rows*planes, cols}.
template <class T>
struct decomposed_fft2 {
  external_plan<T> row;
  std::vector<external_plan<T>> cols;
};

template <class T>
inline decomposed_fft2<T> make_decomposed_fft2(
    std::complex<T> *in, std::complex<T> *work, std::complex<T> *out,
    std::size_t planes, std::size_t rows, std::size_t colsn, int direction,
    unsigned flags) {
  decomposed_fft2<T> d;

  batch_layout rowl;
  rowl.howmany = planes * rows;
  rowl.n = {static_cast<int>(colsn)}; // tight rows: istride 1, idist colsn
  d.row = make_batch_fft_plan(in, work, rowl, direction, flags);

  batch_layout coll;
  coll.howmany = colsn;
  coll.n = {static_cast<int>(rows)};
  coll.istride = static_cast<int>(colsn);
  coll.idist = 1;
  coll.ostride = static_cast<int>(colsn);
  coll.odist = 1;
  const std::size_t plane = rows * colsn;
  d.cols.reserve(planes);
  for (std::size_t p = 0; p < planes; ++p) {
    d.cols.push_back(make_batch_fft_plan(work + p * plane, out + p * plane,
                                         coll, direction, flags));
  }
  return d;
}

// Vector of the transform sizes for a rank-2 or rank-3 layout (either an
// xtensor shape or a caller's int vector).
template <class S>
inline void fft2_sizes(const S &shape, std::size_t &planes, std::size_t &rows,
                       std::size_t &cols) {
  const std::size_t rank = shape.size();
  if (rank != 2 && rank != 3) {
    throw std::invalid_argument(
        "XTENSOR-WRAPPERS: parallel 2-D plans need a rank-2 (single plane) or "
        "rank-3 (planes, rows, cols) input");
  }
  planes = rank == 3 ? static_cast<std::size_t>(shape[0]) : 1;
  rows = static_cast<std::size_t>(shape[rank - 2]);
  cols = static_cast<std::size_t>(shape[rank - 1]);
  if (planes == 0 || rows == 0 || cols == 0) {
    throw std::invalid_argument(
        "XTENSOR-WRAPPERS: 2-D plan dimensions must be non-zero");
  }
}

} // namespace detail

/**
 * @brief Parallel 2-D complex-to-complex transform over an owned output.
 *
 * Runs the separable 2-D DFT as a batched row pass followed by a strided
 * column pass (see the file comment); with OpenMP both passes are executed in
 * parallel over their `howmany` transforms, so a single large 2-D transform
 * uses the whole machine. No transpose is involved and the output is in the
 * native orientation (`output().shape() == input.shape()`).
 *
 * The input array is borrowed and must outlive the plan; the intermediate
 * workspace and the output array are owned by the plan. Move-only; FFTW
 * planning is guarded by the global mutex.
 *
 * @tparam T floating precision (`float` or `double`).
 */
template <class T> class plan_fft2 {
public:
  using output_type = xt::xarray<std::complex<T>>;

  plan_fft2() = default;
  plan_fft2(const plan_fft2 &) = delete;
  plan_fft2 &operator=(const plan_fft2 &) = delete;
  plan_fft2(plan_fft2 &&) noexcept = default;
  plan_fft2 &operator=(plan_fft2 &&) noexcept = default;
  ~plan_fft2() = default;

  /**
   * @brief Creates a parallel 2-D plan from a rank-2 ({rows, cols}) or rank-3
   *        ({planes, rows, cols}) input; one 2-D transform per plane.
   */
  plan_fft2(xt::xarray<std::complex<T>> &input, int direction = FFTW_FORWARD,
            unsigned flags = FFTW_ESTIMATE)
      : m_expected_input_size(input.size()) {
    std::size_t planes, rows, cols;
    detail::fft2_sizes(input.shape(), planes, rows, cols);
    m_output = output_type(input.shape());
    m_work = xt::xarray<std::complex<T>>(
        std::vector<std::size_t>{planes * rows, cols});
    m_impl = detail::make_decomposed_fft2<T>(
        input.data(), m_work.data(), m_output.data(), planes, rows, cols,
        direction, flags);
  }

  /**
   * @brief Executes the transform(s): row pass then column pass.
   *
   * @warning The input array passed to the constructor must remain valid and
   * must not be reallocated until this plan is destroyed.
   */
  void execute() const {
    XTENSOR_FFTW_ASSERT(
        m_expected_input_size != 0 &&
        "execute() on a default-constructed or moved-from plan_fft2");
    m_impl.row.execute();
    for (const auto &c : m_impl.cols) {
      c.execute();
    }
  }

  const output_type &output() const noexcept { return m_output; }

  /**
   * @brief Detaches the output array from the plan. After this, `execute()`
   *        has no valid output buffer, so it must not be called again.
   */
  output_type release() { return std::move(m_output); }

private:
  // Bounded-size scratch; its buffer must be allocated before the row plan is
  // built (the row and column plans borrow raw pointers into it).
  xt::xarray<std::complex<T>> m_work{
      std::vector<std::size_t>{0, 0}};
  detail::decomposed_fft2<T> m_impl;
  output_type m_output;
  std::size_t m_expected_input_size = 0;
};

/**
 * @brief Parallel 2-D complex-to-complex transform over caller-owned buffers.
 *
 * Counterpart of `plan_fft2` for the caller-buffer model: both the input and
 * the output are borrowed raw pointers. The transform is out-of-place (the
 * output must not alias the input). The internal workspace is owned by the
 * plan. Output orientation is native (identical to `plan_fft2`).
 */
template <class T> class external_fft2_plan {
public:
  external_fft2_plan() = default;
  external_fft2_plan(const external_fft2_plan &) = delete;
  external_fft2_plan &operator=(const external_fft2_plan &) = delete;
  external_fft2_plan(external_fft2_plan &&) noexcept = default;
  external_fft2_plan &operator=(external_fft2_plan &&) noexcept = default;
  ~external_fft2_plan() = default;

  /**
   * @brief Creates a parallel 2-D plan over caller buffers.
   *
   * @param in  caller-owned input, `product(n)` complex elements.
   * @param out caller-owned output, same total size; written natively oriented.
   * @param n   transform sizes: 2 ({rows, cols}) or 3 ({planes, rows, cols}).
   */
  external_fft2_plan(std::complex<T> *in, std::complex<T> *out,
                     const std::vector<int> &n, int direction = FFTW_FORWARD,
                     unsigned flags = FFTW_ESTIMATE) {
    std::size_t planes, rows, cols;
    detail::fft2_sizes(std::vector<std::size_t>(n.begin(), n.end()), planes,
                       rows, cols);
    const std::size_t plane = rows * cols;
    m_work = xt::xarray<std::complex<T>>(
        std::vector<std::size_t>{planes * rows, cols});
    m_impl = detail::make_decomposed_fft2<T>(in, m_work.data(), out, planes,
                                             rows, cols, direction, flags);
    m_expected_input_size = planes * plane;
  }

  void execute() const {
    XTENSOR_FFTW_ASSERT(
        m_expected_input_size != 0 &&
        "execute() on a default-constructed or moved-from external_fft2_plan");
    m_impl.row.execute();
    for (const auto &c : m_impl.cols) {
      c.execute();
    }
  }

private:
  xt::xarray<std::complex<T>> m_work{std::vector<std::size_t>{0, 0}};
  detail::decomposed_fft2<T> m_impl;
  std::size_t m_expected_input_size = 0;
};

/**
 * @brief Creates a parallel 2-D c2c plan owning its output (see `plan_fft2`).
 */
template <class T>
inline plan_fft2<T> make_fft2_plan(xt::xarray<std::complex<T>> &input,
                                   int direction = FFTW_FORWARD,
                                   unsigned flags = FFTW_ESTIMATE) {
  return plan_fft2<T>(input, direction, flags);
}

/**
 * @brief Creates a parallel 2-D c2c plan over caller-owned buffers (see
 *        `external_fft2_plan`).
 */
template <class T>
inline external_fft2_plan<T>
make_fft2_plan(std::complex<T> *in, std::complex<T> *out,
               const std::vector<int> &n, int direction = FFTW_FORWARD,
               unsigned flags = FFTW_ESTIMATE) {
  return external_fft2_plan<T>(in, out, n, direction, flags);
}

} // namespace xt::fftw
#endif // XTENSOR_WRAPPERS_PLAN_2D_HPP
