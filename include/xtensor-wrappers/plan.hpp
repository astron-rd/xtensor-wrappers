#ifndef XTENSOR_WRAPPERS_PLAN_HPP
#define XTENSOR_WRAPPERS_PLAN_HPP

// Plan-based FFTW API for xtensor-fftw, split by dimensionality, all in the
// xt::fftw namespace:
//   - plan_1d.hpp: basic_plan<T> and the N-D make_*_plan() factories, including
//     the reusable plan machinery (plan_traits, thread-safe creation).
//   - plan_2d.hpp: plan_fft2<T>/fft2(), a 2D complex FFT decomposed into 1D
//     row/column FFTs built on the plan_1d.hpp machinery.
//   - plan_batch.hpp: batch_plan<T>/make_batch_*_plan(), batched transforms
//   over
//     strided memory via FFTW's guru (plan_many) interface.
//
// Including this umbrella header provides all three. Include a specific header
// instead if only part of it is needed.

#include <xtensor-wrappers/plan_1d.hpp>
#include <xtensor-wrappers/plan_2d.hpp>
#include <xtensor-wrappers/plan_batch.hpp>

#endif // XTENSOR_WRAPPERS_PLAN_HPP
