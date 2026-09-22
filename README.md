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
| `include/xtensor-wrappers/plan.hpp` | Umbrella header: includes `plan_1d.hpp` and `plan_2d.hpp`. |
| `include/xtensor-wrappers/plan_1d.hpp` | Plan-based FFTW API for xtensor-fftw: `basic_plan<T>`, `make_rfft_plan()`, `make_irfft_plan()`, `make_fft_plan()`, `plan_float`/`plan_double`, plus the plan machinery (`plan_traits`, thread-safe creation). N-D, move-only, RAII. `xt::fftw` namespace, so it slots in alongside xtensor-fftw. |
| `include/xtensor-wrappers/plan_2d.hpp` | `plan_fft2<T>` and `fft2()`: 2D complex FFT decomposed into 1D FFTs over rows and columns (separable transform), with a transpose between passes. |

## Using the library

```cpp
#include <xtensor-wrappers/plan.hpp>

// Plan-based N-D FFT; the plan owns its output buffer.
xt::xarray<std::complex<float>> input = ...;
auto plan = xt::fftw::make_fft_plan(input, xt::xarray<std::complex<float>>{});
plan.execute();
// plan.output() holds the result; call plan.execute() again for new input data.

// 2D transform decomposed into 1D row/column FFTs.
auto out2 = xt::fftw::fft2(input2d);
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
- The input array passed to `make_*_plan()`/`plan_fft2` must outlive the plan
  and must not be reallocated while the plan is alive; the output buffer is
  owned by the plan.
