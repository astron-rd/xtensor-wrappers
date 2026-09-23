#ifndef XTENSOR_WRAPPERS_PLAN_BATCH_HPP
#define XTENSOR_WRAPPERS_PLAN_BATCH_HPP

// Batched transforms over strided memory, built directly on FFTW's guru
// interface (fftw_plan_many_*).
//
// A batch is split into contiguous chunks (by default one per OpenMP thread),
// each executed by its own plan_many plan via #pragma omp parallel for over
// the batch items - transparent to the user when XTENSOR_WRAPPERS_USE_OPENMP is
// enabled. Without OpenMP there is exactly one chunk (plain serial execution).
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

#include <algorithm>
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

// Tight input size of one transform for a c2r (irfft) batch, in input
// (half-complex) elements: all dimensions keep their real size except the last,
// which holds n/2 + 1 values.
inline std::size_t half_complex_input_size(const batch_layout &l) {
  std::size_t p = 1;
  for (std::size_t i = 0; i + 1 < l.n.size(); ++i) {
    p *= static_cast<std::size_t>(l.n[i]);
  }
  p *= static_cast<std::size_t>(l.n.back() / 2 + 1);
  return p;
}

// Distance between consecutive c2r (irfft) input transforms, in input
// (half-complex) elements. Unlike the c2c/r2c case the tight default is NOT
// the real transform size: the input rows are half_complex_input_size(l)
// apart, so the plain idist_of() machinery must not be used here.
inline int idist_c2r_of(const batch_layout &l) {
  return l.idist ? static_cast<int>(l.idist)
                 : static_cast<int>(half_complex_input_size(l));
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

// The owning batch factories always produce tightly packed output derived from
// `n`, so a custom output layout (onembed/ostride/odist) cannot be honoured
// there; reject it rather than silently ignoring it.
inline void require_tight_output(const batch_layout &l) {
  if (!l.onembed.empty() || l.ostride != 1 || l.odist != 0) {
    throw std::invalid_argument(
        "XTENSOR-WRAPPERS: owned batch plans always produce tightly packed "
        "output; leave onembed/ostride/odist unset (or use the caller-buffer "
        "overload which writes into your buffer) for padded or strided output");
  }
}

// Number of parallel chunks to split a batch into.
inline std::size_t batch_thread_count() {
#if XTENSOR_WRAPPERS_USE_OPENMP
  return std::max<std::size_t>(1,
                               static_cast<std::size_t>(omp_get_max_threads()));
#else
  return 1;
#endif
}

// Minimum transforms per chunk: below this a chunk degrades into a handful of
// tiny FFTW calls that cannot exploit SIMD, so splitting further only adds
// overhead.
inline constexpr std::size_t min_transforms_per_chunk = 64;

// Number of parallel chunks a batch of `howmany` transforms should be split
// into: at most one per OpenMP thread, and never so many that a chunk falls
// below min_transforms_per_chunk transforms.
inline std::size_t batch_chunk_count(std::size_t howmany,
                                     std::size_t max_chunks = 0) {
  const std::size_t cap =
      max_chunks ? max_chunks : static_cast<std::size_t>(batch_thread_count());
  const std::size_t by_min =
      (howmany + min_transforms_per_chunk - 1) / min_transforms_per_chunk;
  return std::max<std::size_t>(1, std::min(cap, by_min));
}

// Split [0, howmany) into `nthreads` contiguous, balanced chunks (non-empty
// chunks only; empty trailing chunks are dropped).
inline void chunk_ranges(std::size_t howmany, std::size_t nthreads,
                         std::vector<std::size_t> &begins,
                         std::vector<std::size_t> &counts) {
  begins.clear();
  counts.clear();
  const std::size_t per = howmany / nthreads;
  const std::size_t rem = howmany % nthreads;
  std::size_t base = 0;
  for (std::size_t i = 0; i < nthreads; ++i) {
    const std::size_t c = per + (i < rem ? 1 : 0);
    if (c > 0) {
      begins.push_back(base);
      counts.push_back(c);
    }
    base += c;
  }
}

// Build one FFTW plan per contiguous chunk of a batch. Chunk boundary `b`
// starts `b * in_dist` input elements and `b * out_dist` output elements in,
// so every chunk is an independent, contiguous slice of the same buffers.
// The batch is split into batch_chunk_count(howmany) balanced chunks (capped
// so each chunk keeps at least min_transforms_per_chunk transforms).
// `plan_one(rank, howmany, in, out)` issues a single chunk plan with the
// caller's chosen guru routine.
template <class T, class PlanOne>
inline std::vector<typename detail::plan_traits<T>::plan_type>
make_chunked_plans(const batch_layout &l, std::size_t in_dist,
                   std::size_t out_dist, char *in, std::size_t in_elem,
                   char *out, std::size_t out_elem, PlanOne &&plan_one) {
  std::vector<std::size_t> begins, counts;
  chunk_ranges(l.howmany, batch_chunk_count(l.howmany), begins, counts);
  std::vector<typename detail::plan_traits<T>::plan_type> plans;
  plans.reserve(begins.size());
  const int rank = static_cast<int>(l.n.size());
  for (std::size_t i = 0; i < begins.size(); ++i) {
    auto p = plan_one(rank, static_cast<int>(counts[i]),
                      in + begins[i] * in_dist * in_elem,
                      out + begins[i] * out_dist * out_elem);
    if (p == nullptr) {
      throw std::runtime_error(
          "XTENSOR-WRAPPERS: FFTW batch plan creation failed");
    }
    plans.push_back(p);
  }
  return plans;
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
 * The batch is executed as one plan per contiguous chunk; when built with
 * OpenMP the chunks run in parallel (`execute()` is transparently
 * multi-threaded over batch items, each transform single-threaded). Move-only,
 * and FFTW planning is guarded by the global mutex. `execute()` needs no lock
 * (FFTW >= 3.3.5 execution is thread-safe).
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
      : m_plans(std::move(other.m_plans)), m_output(std::move(other.m_output)),
        m_expected_output_size(other.m_expected_output_size) {}

  batch_plan &operator=(batch_plan &&other) noexcept {
    if (this != &other) {
      std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
      destroy();
      m_plans = std::move(other.m_plans);
      m_output = std::move(other.m_output);
      m_expected_output_size = other.m_expected_output_size;
    }
    return *this;
  }

  /**
   * @brief Takes ownership of raw FFTW plan(s) and the output array.
   *
   * @throws std::runtime_error if any plan is null (planner failure).
   */
  explicit batch_plan(std::vector<plan_type> plans, output_type output)
      : m_plans(std::move(plans)), m_output(std::move(output)),
        m_expected_output_size(m_output.size()) {
    for (plan_type p : m_plans) {
      if (p == nullptr) {
        throw std::runtime_error(
            "XTENSOR-WRAPPERS: FFTW batch plan creation failed");
      }
    }
  }

  ~batch_plan() {
    std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
    destroy();
  }

  /**
   * @brief Executes the whole batch of transforms.
   *
   * @warning The input memory passed to the factory must remain valid and must
   * not be moved until this plan is destroyed.
   */
  void execute() const {
    XTENSOR_FFTW_ASSERT(
        !m_plans.empty() &&
        "execute() on a default-constructed or moved-from batch plan");
    XTENSOR_FFTW_ASSERT(m_output.size() == m_expected_output_size &&
                        "output buffer changed since plan creation");
    detail::execute_parallel(m_plans.data(), m_plans.size(),
                             [](plan_type p) { traits::execute(p); });
  }

  /**
   * @brief The first chunk's plan handle (only meaningful for a plan that was
   *        not split across threads).
   */
  plan_type get() const noexcept {
    return m_plans.empty() ? nullptr : m_plans.front();
  }

  const output_type &output() const noexcept { return m_output; }

  /**
   * @brief Detaches the output array from the plan.
   */
  output_type release() { return std::move(m_output); }

private:
  void destroy() noexcept {
    for (plan_type p : m_plans) {
      if (p) {
        traits::destroy_plan(p);
      }
    }
    m_plans.clear();
  }

  std::vector<plan_type> m_plans;
  output_type m_output;
  std::size_t m_expected_output_size = 0;
};

/**
 * @brief Creates a plan for a batch of complex-to-complex FFTs; owns the
 *        output as a tight `{howmany}`-prefixed xarray.
 *
 * @param input borrowed contiguous batch memory (howmany * idist complex
 *        elements).
 * @throws std::invalid_argument if the output layout is padded/strided
 *         (onembed/ostride/odist set): owned output is always tight.
 */
template <class T>
inline batch_plan<T> make_batch_fft_plan(const std::complex<T> *input,
                                         const batch_layout &layout,
                                         int direction = FFTW_FORWARD,
                                         unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  detail::require_tight_output(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/false);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_of(layout);
  const int ostride = 1;
  auto output =
      detail::owned_output<std::complex<T>>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, out_per,
      reinterpret_cast<char *>(const_cast<std::complex<T> *>(input)),
      sizeof(std::complex<T>), reinterpret_cast<char *>(output.data()),
      sizeof(std::complex<T>), [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_c2c(
            rank, layout.n.data(), howmany,
            reinterpret_cast<complex_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<complex_type *>(co),
            out_shape.data(), ostride, out_per, direction, flags);
      });

  return batch_plan<T>(std::move(plans), std::move(output));
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
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_of(layout);
  const int odist = detail::odist_of(layout);
  const int ostride = layout.ostride;

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, odist, reinterpret_cast<char *>(input),
      sizeof(std::complex<T>), reinterpret_cast<char *>(output),
      sizeof(std::complex<T>), [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_c2c(
            rank, layout.n.data(), howmany,
            reinterpret_cast<complex_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<complex_type *>(co),
            layout.onembed.empty() ? nullptr : layout.onembed.data(), ostride,
            odist, direction, flags);
      });

  return external_plan<T>(std::move(plans));
}

/**
 * @brief Creates a plan for a batch of real-to-complex (half-complex) FFTs;
 *        owns the output as a tight `{howmany, ..., n/2+1}` xarray.
 *
 * @throws std::invalid_argument if the output layout is padded/strided
 *         (onembed/ostride/odist set): owned output is always tight.
 */
template <class T>
inline batch_plan<T> make_batch_rfft_plan(const T *input,
                                          const batch_layout &layout,
                                          unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  using real_type = typename traits::real_type;
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  detail::require_tight_output(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/true);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_of(layout);
  const int ostride = 1;
  auto output =
      detail::owned_output<std::complex<T>>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, out_per, reinterpret_cast<char *>(const_cast<T *>(input)),
      sizeof(T), reinterpret_cast<char *>(output.data()),
      sizeof(std::complex<T>), [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_r2c(
            rank, layout.n.data(), howmany, reinterpret_cast<real_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<complex_type *>(co),
            out_shape.data(), ostride, out_per, flags);
      });

  return batch_plan<T>(std::move(plans), std::move(output));
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
  using real_type = typename traits::real_type;
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_of(layout);
  const int odist = detail::odist_of(layout);
  const int ostride = layout.ostride;

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, odist, reinterpret_cast<char *>(input), sizeof(T),
      reinterpret_cast<char *>(output), sizeof(std::complex<T>),
      [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_r2c(
            rank, layout.n.data(), howmany, reinterpret_cast<real_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<complex_type *>(co),
            layout.onembed.empty() ? nullptr : layout.onembed.data(), ostride,
            odist, flags);
      });

  return external_plan<T>(std::move(plans));
}

/**
 * @brief Creates a plan for a batch of complex-to-real (inverse half-complex)
 *        FFTs; owns the output as a tight `{howmany}`-prefixed real xarray.
 *
 * The input is half-complex per transform (e.g. tight `n / 2 + 1` elements, or
 * padded via `inembed`/`idist`); `layout.n` is the real output size.
 *
 * @throws std::invalid_argument if the output layout is padded/strided
 *         (onembed/ostride/odist set): owned output is always tight.
 */
template <class T>
inline batch_plan<T, T> make_batch_irfft_plan(const std::complex<T> *input,
                                              const batch_layout &layout,
                                              unsigned flags = FFTW_ESTIMATE) {
  using traits = detail::plan_traits<T>;
  using real_type = typename traits::real_type;
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  detail::require_tight_output(layout);
  auto out_shape = detail::output_shape(layout, /*half=*/false);
  const int out_per = static_cast<int>(detail::product(out_shape));
  const int idist = detail::idist_c2r_of(layout);
  const int ostride = 1;
  auto output = detail::owned_output<T>(out_shape, layout.howmany);

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, out_per,
      reinterpret_cast<char *>(const_cast<std::complex<T> *>(input)),
      sizeof(std::complex<T>), reinterpret_cast<char *>(output.data()),
      sizeof(T), [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_c2r(
            rank, layout.n.data(), howmany,
            reinterpret_cast<complex_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<real_type *>(co),
            out_shape.data(), ostride, out_per, flags);
      });

  return batch_plan<T, T>(std::move(plans), std::move(output));
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
  using real_type = typename traits::real_type;
  using complex_type = typename traits::complex_type;
  detail::require_valid_layout(layout);
  const int idist = detail::idist_c2r_of(layout);
  const int odist = detail::odist_of(layout);
  const int ostride = layout.ostride;

  std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
  auto plans = detail::make_chunked_plans<T>(
      layout, idist, odist, reinterpret_cast<char *>(input),
      sizeof(std::complex<T>), reinterpret_cast<char *>(output), sizeof(T),
      [&](int rank, int howmany, char *ci, char *co) {
        return traits::make_many_c2r(
            rank, layout.n.data(), howmany,
            reinterpret_cast<complex_type *>(ci),
            layout.inembed.empty() ? nullptr : layout.inembed.data(),
            layout.istride, idist, reinterpret_cast<real_type *>(co),
            layout.onembed.empty() ? nullptr : layout.onembed.data(), ostride,
            odist, flags);
      });

  return external_plan<T, T>(std::move(plans));
}

} // namespace xt::fftw
#endif // XTENSOR_WRAPPERS_PLAN_BATCH_HPP
