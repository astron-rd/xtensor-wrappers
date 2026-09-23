#ifndef XTENSOR_WRAPPERS_PLAN_HPP
#define XTENSOR_WRAPPERS_PLAN_HPP

// Plan-based FFTW API for xtensor-fftw, all in the xt::fftw namespace:
//   - plan_1d.hpp: basic_plan<T>/external_plan<T> and the N-D make_*_plan()
//     factories, including the reusable plan machinery (plan_traits,
//     thread-safe creation). Two factory overloads select the buffer model:
//     xtensor inputs own their output (basic_plan), raw pointers transform
//     between caller buffers (external_plan). Rank-generic, so 2D is covered
//     by the same API as 1D.
//   - plan_batch.hpp: batch_plan<T>/make_batch_*_plan(), batched transforms
//     over strided memory via FFTW's guru (plan_many) interface, with the same
//     owning / caller-buffer overload split.
//   - plan_2d.hpp: plan_fft2<T>/external_fft2_plan<T>, a parallel 2-D c2c
//     transform (batched row + strided column passes) for few large
//     transformations.
//
// Including this umbrella header provides all three. Include a specific header
// instead if only part of it is needed.

#include <xtensor-wrappers/plan_1d.hpp>
#include <xtensor-wrappers/plan_2d.hpp>
#include <xtensor-wrappers/plan_batch.hpp>

#endif // XTENSOR_WRAPPERS_PLAN_HPP
