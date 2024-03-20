# Design

This page explains the default implementations and design choices of AbstractMCMC.
It is not intended for users but for developers that want to implement the AbstractMCMC
interface for Markov chain Monte Carlo sampling. The user-facing API is explained in
[API](@ref).

## Overview

AbstractMCMC provides a default implementation of the user-facing interface described
in [API](@ref). You can completely neglect these and define your own implementation of the
interface. However, as described below, in most use cases the default implementation
allows you to obtain support of parallel sampling, progress logging, callbacks, iterators,
and transducers for free by just defining the sampling step of your inference algorithm,
drastically reducing the amount of code you have to write. In general, the docstrings
of the functions described below might be helpful if you intend to make use of the default
implementations.

## Basic structure

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

## Sampling step

The only method for which no default implementation is provided (and hence which
downstream packages *have* to implement) is [`AbstractMCMC.step`](@ref). It defines
the sampling step of the inference method.

```@docs
AbstractMCMC.step
```

If one also has some special handling of the warmup-stage of sampling, then this can be specified by overloading

```@docs
AbstractMCMC.step_warmup
```

which will be used for the first `num_warmup` iterations, as specified as a keyword argument to [`AbstractMCMC.sample`](@ref). 
Note that this is optional; by default it simply calls [`AbstractMCMC.step`](@ref) from above.

## Collecting samples

!!! note
    This section does not apply to the iterator and transducer interface.

After the initial sample is obtained, the default implementations for regular and parallel sampling
(not for the iterator and the transducer since it is not needed there) create a container for all
samples (the initial one and all subsequent samples) using `AbstractMCMC.samples`.

```@docs
AbstractMCMC.samples
```

In each step, the sample is saved in the container by `AbstractMCMC.save!!`. The notation `!!`
follows the convention of the package [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl)
which is used in the default implementation of `AbstractMCMC.save!!`. It indicates that the
sample is pushed to the container but a "widening" fallback is used if the container type
does not allow to save the sample. Therefore `AbstractMCMC.save!!` *always has* to return the container.

```@docs
AbstractMCMC.save!!
```

For most use cases the default implementation of `AbstractMCMC.samples` and `AbstractMCMC.save!!`
should work out of the box and hence need not to be overloaded in downstream code.

## Creating chains

!!! note
    This section does not apply to the iterator and transducer interface.

At the end of the sampling procedure for regular and paralle sampling we transform
the collection of samples to the desired output type by calling `AbstractMCMC.bundle_samples`.

```@docs
AbstractMCMC.bundle_samples
```

The default implementation should be fine in most use cases, but downstream packages
could, e.g., save the final state of the sampler as well if they overload
`AbstractMCMC.bundle_samples`.
