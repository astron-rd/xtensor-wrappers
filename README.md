# xtensor-wrappers

Header-only utility library for working with [xtensor](https://github.com/xtensor-stack/xtensor)
and [xtensor-fftw](https://github.com/xtensor-stack/xtensor-fftw), providing a
reusable plan-based FFTW API and higher-level wrappers built on it.

The xtensor stack (including xtensor-fftw 0.4.0) is fetched automatically by
CMake; FFTW is taken from the provided Spack modules. No performance
benchmarks are included at this stage — the tests check correctness against a
raw FFTW reference.

## Requirements

The user is assumed to have loaded the Spack environment modules:

```
module load spack/2026.06.23 fftw/3.3.11 hdf5/1.14.6 gsl/2.8
```

(only `fftw` is strictly required; `gsl`/`hdf5` are loaded for consistency with
sibling projects).

## Build and test

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Contents

| Header | Purpose |
| ------ | ------- |
| `include/xtensor-wrappers/plan.hpp` | Umbrella header: includes `plan_1d.hpp` and `plan_batch.hpp`. |
| `include/xtensor-wrappers/plan_1d.hpp` | Plan-based FFTW API for xtensor-fftw: `basic_plan<T>` (owns its output) and `external_plan<T>` (transforms between caller-owned buffers), `make_rfft_plan()`/`make_irfft_plan()`/`make_fft_plan()`, `plan_float`/`plan_double`, plus the plan machinery (`plan_traits`, thread-safe creation). Each factory has two overloads selecting the buffer model: an xtensor input owns its output (`basic_plan`), raw pointers transform between caller buffers (`external_plan`, in-place allowed). Rank-generic — 1D through N-D, 2D included — and move-only RAII in the `xt::fftw` namespace. |
| `include/xtensor-wrappers/plan_batch.hpp` | `batch_plan<T>`/`make_batch_*_plan()`: howmany identical transforms over strided memory via FFTW's guru (`plan_many`) interface, with per-direction padding/strides (`batch_layout`). The owning overloads return a tight `{howmany}`-prefixed `xt::xarray`; the overload taking an output pointer writes into caller-owned buffers (honouring `onembed`/`ostride`/`odist`). |

## Using the library

```cpp
#include <xtensor-wrappers/plan.hpp>

// Plan-based N-D FFT; the plan owns its output buffer (xtensor overload).
xt::xarray<std::complex<float>> input = ...;
auto plan = xt::fftw::make_fft_plan(input, xt::xarray<std::complex<float>>{});
plan.execute();
// plan.output() holds the result; call plan.execute() again for new input data,
// e.g. a 2D input uses the same rank-generic API.

// Same transform into caller-owned buffers (pointer overload; 2D shown).
std::vector<std::complex<float>> out(m * n);
auto p = xt::fftw::make_fft_plan(input.data(), out.data(), {m, n});
p.execute();

// Batch: k transforms of length n on padded input (1000 within 1024) that
// owns a {k, n} output array.
xt::fftw::batch_layout layout;
layout.howmany = k;
layout.n = {1000};
layout.inembed = {1024};
layout.idist = 1024;
auto bp = xt::fftw::make_batch_fft_plan(in_data, layout);
bp.execute(); // bp.output() has shape {k, 1000}
```

Consumers link the target and get the whole dependency tree:

```cmake
add_library(myapp ...)
target_link_libraries(myapp xtensor-wrappers)
```

## Guarantees

- Plan creation and destruction are guarded by a global mutex (FFTW planning is
  not thread-safe); execution needs no lock (FFTW >= 3.3.5).
- FFTW plan-creation failure is reported at runtime (`std::runtime_error`), not
  via assertions that compile out of Release builds.
- The input array passed to `make_*_plan()` must outlive the plan
  and must not be reallocated while the plan is alive. In the xtensor overloads
  the output buffer is owned by the plan; in the pointer overloads the caller
  owns both buffers (`external_plan`, in-place allowed).

## Notes

- `plan_float` / `plan_double` are convenience aliases for `basic_plan<float>` /
  `basic_plan<double>`. Equivalent aliases for `external_plan`/`batch_plan` may
  be added later; today you write the template parameter explicitly.
- `batch_plan::output()` is an `xt::xarray` shaped `{howmany}` followed by the
  per-transform output shape (e.g. `{k, n}` for batched 1D c2c), always tightly
  packed and contiguous. Padded/strided *output* is only available through the
  caller-buffer overloads (which honour `onembed`/`ostride`/`odist`); padded
  *input* works in both (`inembed`/`idist`/`istride`).
