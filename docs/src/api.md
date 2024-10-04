# API

AbstractMCMC defines an interface for sampling Markov chains.

## Model

```@docs
AbstractMCMC.AbstractModel
AbstractMCMC.LogDensityModel
```

## Sampler

```@docs
AbstractMCMC.AbstractSampler
```

## Sampling a single chain

```@docs
AbstractMCMC.sample(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler, ::Any)
AbstractMCMC.sample(::AbstractRNG, ::Any, ::AbstractMCMC.AbstractSampler, ::Any)

```

### Iterator

```@docs
AbstractMCMC.steps(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler)
AbstractMCMC.steps(::AbstractRNG, ::Any, ::AbstractMCMC.AbstractSampler)
```

### Transducer

```@docs
AbstractMCMC.Sample(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler)
AbstractMCMC.Sample(::AbstractRNG, ::Any, ::AbstractMCMC.AbstractSampler)
```

## Sampling multiple chains in parallel

```@docs
AbstractMCMC.sample(
    ::AbstractRNG,
    ::AbstractMCMC.AbstractModel,
    ::AbstractMCMC.AbstractSampler,
    ::AbstractMCMC.AbstractMCMCEnsemble,
    ::Integer,
    ::Integer,
)
AbstractMCMC.sample(
    ::AbstractRNG,
    ::Any,
    ::AbstractMCMC.AbstractSampler,
    ::AbstractMCMC.AbstractMCMCEnsemble,
    ::Integer,
    ::Integer,
)
```

Two algorithms are provided for parallel sampling with multiple threads and multiple processes, and one allows for the user to sample multiple chains in serial (no parallelization):
```@docs
AbstractMCMC.MCMCThreads
AbstractMCMC.MCMCDistributed
AbstractMCMC.MCMCSerial
```

## Common keyword arguments

Common keyword arguments for regular and parallel sampling are:
- `progress` (default: `AbstractMCMC.PROGRESS[]` which is `true` initially):  toggles progress logging
- `chain_type` (default: `Any`): determines the type of the returned chain
- `callback` (default: `nothing`): if `callback !== nothing`, then
  `callback(rng, model, sampler, sample, iteration)` is called after every sampling step,
  where `sample` is the most recent sample of the Markov chain and `iteration` is the current iteration
- `num_warmup` (default: `0`): number of "warm-up" steps to take before the first "regular" step, 
   i.e. number of times to call [`AbstractMCMC.step_warmup`](@ref) before the first call to 
   [`AbstractMCMC.step`](@ref).
- `discard_initial` (default: `num_warmup`): number of initial samples that are discarded. Note that
  if `discard_initial < num_warmup`, warm-up samples will also be included in the resulting samples.
- `thinning` (default: `1`): factor by which to thin samples.
- `initial_state` (default: `nothing`): if `initial_state !== nothing`, the first call to [`AbstractMCMC.step`](@ref)
  is passed `initial_state` as the `state` argument.

!!! info
    The common keyword arguments `progress`, `chain_type`, and `callback` are not supported by the iterator [`AbstractMCMC.steps`](@ref) and the transducer [`AbstractMCMC.Sample`](@ref).

There is no "official" way for providing initial parameter values yet.
However, multiple packages such as [EllipticalSliceSampling.jl](https://github.com/TuringLang/EllipticalSliceSampling.jl) and [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) support an `initial_params` keyword argument for setting the initial values when sampling a single chain.
To ensure that sampling multiple chains "just works" when sampling of a single chain is implemented, [we decided to support `initial_params` in the default implementations of the ensemble methods](https://github.com/TuringLang/AbstractMCMC.jl/pull/94):
- `initial_params` (default: `nothing`): if `initial_params isa AbstractArray`, then the `i`th element of `initial_params` is used as initial parameters of the `i`th chain. If one wants to use the same initial parameters `x` for every chain, one can specify e.g. `initial_params = FillArrays.Fill(x, N)`.

Progress logging can be enabled and disabled globally with `AbstractMCMC.setprogress!(progress)`.

```@docs
AbstractMCMC.setprogress!
```

## Chains

The `chain_type` keyword argument allows to set the type of the returned chain. A common
choice is to return chains of type `Chains` from [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl).

AbstractMCMC defines the abstract type `AbstractChains` for Markov chains.

```@docs
AbstractMCMC.AbstractChains
```

For chains of this type, AbstractMCMC defines the following two methods.

```@docs
AbstractMCMC.chainscat
AbstractMCMC.chainsstack
```
