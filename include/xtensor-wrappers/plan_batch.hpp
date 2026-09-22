#ifndef XTENSOR_WRAPPERS_PLAN_BATCH_HPP
#define XTENSOR_WRAPPERS_PLAN_BATCH_HPP

// Batched transforms over strided memory, built directly on FFTW's guru
// interface (fftw_plan_many_*). One FFTW plan executes all `howmany`
// transforms, which lets FFTW vectorize across the batch internally.
//
// Two factory overloads provide the two buffer models, mirroring the single
// transform API:
//   - make_batch_*_plan(input, layout, ...): the plan OWNS its output, an
//     xt::xarray of shape `{howmany}` followed by the per-transform output
//     shape, always tightly packed and contiguous.
//   - make_batch_*_plan(input, output, layout, ...): the plan writes into a
//     caller-owned output buffer (returns an external_plan); the output
//     layout in batch_layout (onembed/ostride/odist) is honored exactly.
//
// The input side is always strided/padded via batch_layout (inembed/idist/
// istride), e.g. a 1000-point FFT inside a 1024-element allocation.

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xarray.hpp>

#include <xtensor-wrappers/plan_1d.hpp>

namespace xt::fftw {

/**
 * @brief Describes a batch of identical transforms over strided memory.
 *
 * Maps one-to-one onto FFTW's guru (plan_many) interface: `howmany` identical
 * transforms of `n` are applied to buffers whose allocated size is `inembed`
 * (per dimension) and whose elements are `istride` apart, with consecutive
 * transforms `idist` apart. A `dist` of 0 or an empty `embed` means "tightly
 * packed", i.e. derived from the transform size.
 *
 * The input side (inembed/istride/idist) is always honored. The output side
 * (onembed/ostride/odist) is honored by the borrowing factory overloads; the
 * owning overloads ignore it because they always return tightly packed output.
 */
struct batch_layout {
  std::size_t howmany = 1;
  std::vector<int> n;       // transform size(s)
  std::vector<int> inembed; // allocated input size per dim; empty => tight
  int istride = 1;          // element stride within a transform
  std::size_t idist = 0;    // element distance between transforms; 0 => tight
  std::vector<int> onembed; // allocated output size per dim; empty => tight
  int ostride = 1;
  std::size_t odist = 0; // 0 => tight
};

namespace detail {

inline std::size_t product(const std::vector<int> &v) {
  std::size_t p = 1;
  for (int e : v) {
    p *= static_cast<std::size_t>(e);
  }
  return p;
}

// Elements per transform for the given (possibly padded) embed, falling back
// to the transform size when tightly packed.
inline std::size_t elements_per_transform(const batch_layout &l,
                                          const std::vector<int> &embed) {
  return embed.empty() ? product(l.n) : product(embed);
}

// Distance between consecutive input transforms, in elements.
inline int idist_of(const batch_layout &l) {
  return l.idist ? static_cast<int>(l.idist)
                 : static_cast<int>(elements_per_transform(l, l.inembed));
}

// Distance between consecutive output transforms, in elements (honours the
// caller's output padding; used by the borrowing factories).
inline int odist_of(const batch_layout &l) {
  return l.odist ? static_cast<int>(l.odist)
                 : static_cast<int>(elements_per_transform(l, l.onembed));
}

// Tight logical output shape of one transform: `l.n`, with the last dimension
// turned into the r2c half-complex size when `half` is set.
inline std::vector<int> output_shape(const batch_layout &l, bool half) {
  std::vector<int> s = l.n;
  if (half) {
    s.back() = s.back() / 2 + 1;
  }
  return s;
}

// Build the owned output array: {howmany} prepended to `shape`.
template <class StoredT>
inline xt::xarray<StoredT> owned_output(const std::vector<int> &shape,
                                        std::size_t howmany) {
  std::vector<std::size_t> full;
  full.reserve(shape.size() + 1);
  full.push_back(howmany);
  for (int e : shape) {
    full.push_back(static_cast<std::size_t>(e));
  }
  return xt::xarray<StoredT>(std::move(full));
}

inline void require_valid_layout(const batch_layout &l) {
  if (l.howmany == 0) {
    throw std::invalid_argument(
        "XTENSOR-WRAPPERS: batch_layout.howmany must be > 0");
  }
  if (l.n.empty()) {
    throw std::invalid_argument(
        "XTENSOR-WRAPPERS: batch_layout.n must not be empty");
  }
  for (int dim : l.n) {
    if (dim <= 0) {
      throw std::invalid_argument(
          "XTENSOR-WRAPPERS: batch_layout.n contains a non-positive size");
    }
  }
}

} // namespace detail

/**
 * @brief RAII wrapper around a batched FFTW plan (guru plan_many).
 *
 * Mirrors `basic_plan`'s ownership contract: the output is owned by the plan
 * as an `xt::xarray<StoredT>` of shape `{howmany}` followed by the
 * per-transform output shape (always tightly packed and contiguous, e.g.
 * `{howmany, n}` for a batch of 1D c2c). The *input* is borrowed from the
 * caller, so it must outlive the plan and must not be reallocated while the
 * plan is alive.
 *
 * Move-only, and FFTW planning is guarded by the global mutex. `execute()`
 * needs no lock (FFTW >= 3.3.5 execution is thread-safe).
 *
 * @tparam T floating precision (`float` or `double`).
 * @tparam StoredT element type of the owned output array: `std::complex<T>`
 *         for C2C/R2C plans, `T` for C2R plans.
 */
template <class T, class StoredT = std::complex<T>> class batch_plan {
public:
  using traits = detail::plan_traits<T>;
  using plan_type = typename traits::plan_type;
  using output_type = xt::xarray<StoredT>;

  batch_plan() = default;

  batch_plan(const batch_plan &) = delete;
  batch_plan &operator=(const batch_plan &) = delete;

  batch_plan(batch_plan &&other) noexcept
      : m_plan(other.m_plan), m_output(std::move(other.m_output)),
        m_expected_output_size(other.m_expected_output_size) {
    other.m_plan = nullptr;
  }

  batch_plan &operator=(batch_plan &&other) noexcept {
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
   * @brief Takes ownership of a raw batched FFTW plan and its output array.
   *
   * @throws std::runtime_error if `p` is null (planner failure).
   */
  explicit batch_plan(plan_type p, output_type output)
      : m_plan(p), m_output(std::move(output)),
        m_expected_output_size(m_output.size()) {
    if (p == nullptr) {
      throw std::runtime_error(
          "XTENSOR-WRAPPERS: FFTW batch plan creation failed");
    }
  }

  ~batch_plan() {
    if (m_plan) {
      std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
      traits::destroy_plan(m_plan);
    }
  }

  /**
   * @brief Executes the whole batch of transforms.
   *
   * @warning The input memory passed to the factory must remain valid and must
   * not be moved until this plan is destroyed.
   */
  void execute() const {
    XTENSOR_FFTW_ASSERT(
        m_plan != nullptr &&
        "execute() on a default-constructed or moved-from batch plan");
    XTENSOR_FFTW_ASSERT(m_output.size() == m_expected_output_size &&
                        "output buffer changed since plan creation");
    traits::execute(m_plan);
  }

  plan_type get() const noexcept { return m_plan; }

  const output_type &output() const noexcept { return m_output; }

  /**
   * @brief Detaches the output array from the plan.
   */
  output_type release() { return std::move(m_output); }

private:
  plan_type m_plan = nullptr;
  output_type m_output;
  std::size_t m_expected_output_size = 0;
};

/**
 * @brief Creates a plan for a batch of complex-to-complex FFTs; owns the
 *        output as a tight `{howmany}`-prefixed xarray.
 *
 * @param input borrowed contiguous batch memory (howmany * idist complex
 *        elements).
 */
template <class T>
inline batch_plan<T> make_batch_fft_plan(const std::complex<T> *input,
                                         const batch_layout &layout,
                                         int direction = FFTW_FORWARD,
                                         unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/false);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_of(layout);

  auto output =
      detail::owned_output<std::complex<T>>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_c2c(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany),
      reinterpret_cast<typename traits::complex_type *>(
          const_cast<std::complex<T> *>(input)),
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, reinterpret_cast<typename traits::complex_type *>(output.data()),
      out_shape.data(), 1, out_per, direction, flags);

  return batch_plan<T>(p, std::move(output));
}

/**
 * @brief Creates a batched complex-to-complex plan over caller-owned buffers.
 *
 * Overload of `make_batch_fft_plan` that writes into the caller's `output`
 * buffer instead of an owned one (returns an `external_plan`); the output
 * layout is governed by `onembed`/`ostride`/`odist` and must fit the caller's
 * buffer. `output == input` performs an in-place batch.
 */
template <class T>
inline external_plan<T>
make_batch_fft_plan(std::complex<T> *input, std::complex<T> *output,
                    const batch_layout &layout, int direction = FFTW_FORWARD,
                    unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_of(layout);
  const int odist = detail::odist_of(layout);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_c2c(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany),
      reinterpret_cast<typename traits::complex_type *>(input),
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, reinterpret_cast<typename traits::complex_type *>(output),
      layout.onembed.empty() ? nullptr : layout.onembed.data(), layout.ostride,
      odist, direction, flags);

  return external_plan<T>(p);
}

/**
 * @brief Creates a plan for a batch of real-to-complex (half-complex) FFTs;
 *        owns the output as a tight `{howmany, ..., n/2+1}` xarray.
 */
template <class T>
inline batch_plan<T> make_batch_rfft_plan(const T *input,
                                          const batch_layout &layout,
                                          unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/true);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_of(layout);

  auto output =
      detail::owned_output<std::complex<T>>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_r2c(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany), const_cast<T *>(input),
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, reinterpret_cast<typename traits::complex_type *>(output.data()),
      out_shape.data(), 1, out_per, flags);

  return batch_plan<T>(p, std::move(output));
}

/**
 * @brief Creates a batched real-to-complex plan over caller-owned buffers.
 *
 * Overload of `make_batch_rfft_plan` that writes into the caller's `output`
 * buffer (returns an `external_plan`); the output is half-complex per
 * transform as governed by `onembed`/`ostride`/`odist`.
 */
template <class T>
inline external_plan<T> make_batch_rfft_plan(T *input, std::complex<T> *output,
                                             const batch_layout &layout,
                                             unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_of(layout);
  const int odist = detail::odist_of(layout);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_r2c(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany), input,
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, reinterpret_cast<typename traits::complex_type *>(output),
      layout.onembed.empty() ? nullptr : layout.onembed.data(), layout.ostride,
      odist, flags);

  return external_plan<T>(p);
}

/**
 * @brief Creates a plan for a batch of complex-to-real (inverse half-complex)
 *        FFTs; owns the output as a tight `{howmany}`-prefixed real xarray.
 *
 * The input is half-complex per transform (e.g. tight `n / 2 + 1` elements, or
 * padded via `inembed`/`idist`); `layout.n` is the real output size.
 */
template <class T>
inline batch_plan<T, T> make_batch_irfft_plan(const std::complex<T> *input,
                                              const batch_layout &layout,
                                              unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/false);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_of(layout);

  auto output = detail::owned_output<T>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_c2r(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany),
      reinterpret_cast<typename traits::complex_type *>(
          const_cast<std::complex<T> *>(input)),
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, output.data(), out_shape.data(), 1, out_per, flags);

  return batch_plan<T, T>(p, std::move(output));
}

/**
 * @brief Creates a batched complex-to-real plan over caller-owned buffers.
 *
 * Overload of `make_batch_irfft_plan` that writes into the caller's `output`
 * buffer (returns an `external_plan`); the output layout is governed by
 * `onembed`/`ostride`/`odist`.
 */
template <class T>
inline external_plan<T, T>
make_batch_irfft_plan(std::complex<T> *input, T *output,
                      const batch_layout &layout,
                      unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_of(layout);
  const int odist = detail::odist_of(layout);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto p = traits::make_many_c2r(
      static_cast<int>(layout.n.size()), layout.n.data(),
      static_cast<int>(layout.howmany),
      reinterpret_cast<typename traits::complex_type *>(input),
      layout.inembed.empty() ? nullptr : layout.inembed.data(), layout.istride,
      idist, output, layout.onembed.empty() ? nullptr : layout.onembed.data(),
      layout.ostride, odist, flags);

  return external_plan<T, T>(p);
}

} // namespace xt::fftw
#endif // XTENSOR_WRAPPERS_PLAN_BATCH_HPP
