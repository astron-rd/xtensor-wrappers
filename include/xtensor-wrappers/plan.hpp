#ifndef XTENSOR_WRAPPERS_PLAN_HPP
#define XTENSOR_WRAPPERS_PLAN_HPP

// Plan-based FFTW API for xtensor-fftw, split by dimensionality, all in the
// xt::fftw namespace:
//   - plan_1d.hpp: basic_plan<T> and the N-D make_*_plan() factories, including
//     the reusable plan machinery (plan_traits, thread-safe creation).
//   - plan_2d.hpp: plan_fft2<T>/fft2(), a 2D complex FFT decomposed into 1D
//     row/column FFTs built on the plan_1d.hpp machinery.
//
// Including this umbrella header provides both. Include the specific header
// instead if only one part is needed.

#include <xtensor-wrappers/plan_1d.hpp>
#include <xtensor-wrappers/plan_2d.hpp>

#endif // XTENSOR_WRAPPERS_PLAN_HPP
