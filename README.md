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
| `include/xtensor-wrappers/plan_1d.hpp` | Plan-based FFTW API for xtensor-fftw: `basic_plan<T>` (owns its output) and `external_plan<T>` (transforms between caller-owned buffers), `make_rfft_plan()`/`make_irfft_plan()`/`make_fft_plan()` and their `*_into()` counterparts, `plan_float`/`plan_double`, plus the plan machinery (`plan_traits`, thread-safe creation). Rank-generic — 1D through N-D, 2D included — and move-only RAII in the `xt::fftw` namespace. |
| `include/xtensor-wrappers/plan_batch.hpp` | `batch_plan<T>`/`make_batch_*_plan()`: howmany identical transforms over strided memory via FFTW's guru (`plan_many`) interface, with per-direction padding/strides (`batch_layout`) and owned-output plus `*_into` caller-buffer variants. |

## Using the library

```cpp
#include <xtensor-wrappers/plan.hpp>

// Plan-based N-D FFT; the plan owns its output buffer.
xt::xarray<std::complex<float>> input = ...;
auto plan = xt::fftw::make_fft_plan(input, xt::xarray<std::complex<float>>{});
plan.execute();
// plan.output() holds the result; call plan.execute() again for new input data.

// Same transform into caller-owned buffers (2D shown; rank-generic).
std::vector<std::complex<float>> out(m * n);
auto p = xt::fftw::make_fft_plan_into(input.data(), out.data(), {m, n});
p.execute();
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
  and must not be reallocated while the plan is alive; the output buffer is
  owned by the plan (borrowing variants: `external_plan`, `*_into`).
