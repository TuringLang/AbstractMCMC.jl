# AbstractMCMC.jl

Abstract types and interfaces for Markov chain Monte Carlo methods.

[![Build Status](https://travis-ci.com/TuringLang/AbstractMCMC.jl.svg?branch=master)](https://travis-ci.com/TuringLang/EllipticalSliceSampling.jl)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractMCMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractMCMC.jl)
[![Coveralls](https://coveralls.io/repos/github/TuringLang/AbstractMCMC.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/AbstractMCMC.jl?branch=master)

## Overview

AbstractMCMC defines an interface for sampling and combining Markov chains.
It comes with a default sampling algorithm that provides support of progress
bars, parallel sampling (multithreaded and multicore), and user-provided callbacks
out of the box. Typically developers only have to define the sampling step
of their inference method in an iterator-like fashion to make use of this
functionality. Additionally, the package defines an iterator and a transducer
for sampling Markov chains based on the interface.

## User-facing API

The user-facing sampling API consists of
```julia
StatsBase.sample(
    [rng::Random.AbstractRNG,]
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    nsamples[;
    kwargs...]
)
```
and
```julia
StatsBase.sample(
    [rng::Random.AbstractRNG,]
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    parallel::AbstractMCMC.AbstractMCMCParallel,
    nsamples::Integer,
    nchains::Integer[;
    kwargs...]
)
```
for regular and parallel sampling, respectively. In regular sampling, users may
provide a function
```julia
isdone(rng, model, sampler, samples, iteration; kwargs...)
```
that returns `true` when sampling should end, and `false` otherwise, instead of
a fixed number of samples `nsamples`. AbstractMCMC defines the abstract types
`AbstractMCMC.AbstractModel`, `AbstractMCMC.AbstractSampler`, and
`AbstractMCMC.AbstractMCMCParallel` for models, samplers, and parallel sampling
algorithms, respectively. Two algorithms `MCMCThreads` and `MCMCDistributed`
are provided for parallel sampling with multiple threads and multiple processes,
respectively.

The function
```julia
AbstractMCMC.steps([rng::AbstractRNG, ]model::AbstractModel, sampler::AbstractSampler[; kwargs...])
```
returns an iterator that returns samples continuously, without a predefined
stopping condition. Similarly,
```julia
AbstractMCMC.Sample([rng::Random.AbstractRNG, ]model::AbstractModel, sampler::AbstractSampler[; kwargs...])
```
returns a transducer that returns samples continuously.

Common keyword arguments for regular and parallel sampling (not supported by the iterator and transducer)
are:
- `progress` (default: `true`):  toggles progress logging
- `chain_type` (default: `Any`): determines the type of the returned chain
- `callback` (default: `nothing`): if `callback !== nothing`, then
  `callback(rng, model, sampler, sample, iteration)` is called after every sampling step,
  where `sample` is the most recent sample of the Markov chain and `iteration` is the current iteration
- `discard_initial` (default: `0`): number of initial samples that are discarded
- `thinning` (default: `1`): factor by which to thin samples.

Additionally, AbstractMCMC defines the abstract type `AbstractChains` for Markov chains and the
method `AbstractMCMC.chainscat(::AbstractChains...)` for concatenating multiple chains.
(defaults to `cat(::AbstractChains...; dims = 3)`).

Note that AbstractMCMC exports only `MCMCThreads` and `MCMCDistributed` (and in
particular not `StatsBase.sample`).

## Developer documentation: Default implementation

AbstractMCMC provides a default implementation of the user-facing interface described
above. You can completely neglect these and define your own implementation of the
interface. However, as described below, in most use cases the default implementation
allows you to obtain support of parallel sampling, progress logging, callbacks, iterators,
and transducers for free by just defining the sampling step of your inference algorithm,
drastically reducing the amount of code you have to write. In general, the docstrings
of the functions described below might be helpful if you intend to make use of the default
implementations.

### Basic structure

The simplified structure for regular sampling (the actual implementation contains
some additional error checks and support for progress logging and callbacks) is
```julia
StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    nsamples::Integer;
    chain_type = ::Type{Any},
    kwargs...
)
    # Obtain the initial sample and state.
    sample, state = AbstractMCMC.step(rng, model, sampler; kwargs...)

    # Save the sample.
    samples = AbstractMCMC.samples(sample, model, sampler, N; kwargs...)
    samples = AbstractMCMC.save!!(samples, sample, 1, model, sampler, N; kwargs...)

    # Step through the sampler.
    for i in 2:N
        # Obtain the next sample and state.
        sample, state = AbstractMCMC.step(rng, model, sampler, state; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.save!!(samples, sample, i, model, sampler, N; kwargs...)
    end
    
    return AbstractMCMC.bundle_samples(samples, model, sampler, state, chain_type; kwargs...)
end
```
All other default implementations make use of the same structure and in particular
call the same methods.

### Sampling step

The only method for which no default implementation is provided (and hence which
downstream packages *have* to implement) is `AbstractMCMC.step`
that defines the sampling step of the inference method. In the initial step it is
called as
```julia
AbstractMCMC.step(rng, model, sampler; kwargs...)
```
whereas in all subsequent steps it is called as
```julia
AbstractMCMC.step(rng, model, sampler, state; kwargs...)
```
where `state` denotes the current state of the sampling algorithm. It should return
a 2-tuple consisting of the next sample and the updated state of the sampling algorithm.
Hence `AbstractMCMC.step` can be viewed as an extended version of
[`Base.iterate`](https://docs.julialang.org/en/v1/base/collections/#lib-collections-iteration-1)
with additional positional and keyword arguments.

### Collecting samples (does not apply to the iterator and transducer)

After the initial sample is obtained, the default implementations for regular and parallel sampling
(not for the iterator and the transducer since it is not needed there) create a container for all
samples (the initial one and all subsequent samples) using `AbstractMCMC.samples`. By default,
`AbstractMCMC.samples` just returns a concretely typed `Vector` with the initial sample as single
entry. If the total number of samples is fixed, we use `sizehint!` to suggest that the container
reserves capacity for all samples to improve performance.

In each step, the sample is saved in the container by `AbstractMCMC.save!!`. The notation `!!`
follows the convention of the package [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl)
which is used in the default implementation of `AbstractMCMC.save!!`. It indicates that the
sample is pushed to the container but a "widening" fallback is used if the container type
does not allow to save the sample. Therefore `AbstractMCMC.save!!` *always has* to return the container.

For most use cases the default implementation of `AbstractMCMC.samples` and `AbstractMCMC.save!!`
should work out of the box and hence need not to be overloaded in downstream code. Please have
a look at the docstrings of `AbstractMCMC.samples` and `AbstractMCMC.save!!` if you intend
to overload these functions.

### Creating chains (does not apply to the iterator and transducer)

At the end of the sampling procedure for regular and paralle sampling (not for the iterator
and the transducer) we transform the collection of samples to the desired output type by
calling
```julia
AbstractMCMC.bundle_samples(samples, model, sampler, state, chain_type; kwargs...)
```
where `samples` is the collection of samples, `state` is the final state of the sampler,
and `chain_type` is the desired return type. The default implementation in AbstractMCMC
just returns the collection `samples`.

The default implementation should be fine in most use cases, but downstream packages
could, e.g., save the final state of the sampler as well if they overload `AbstractMCMC.bundle_samples`.
