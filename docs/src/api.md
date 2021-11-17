# API

AbstractMCMC defines an interface for sampling Markov chains.

## Sampling a single chain

```@docs
AbstractMCMC.sample(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler, ::Integer)
AbstractMCMC.sample(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler, ::Any)
```

### Iterator

```@docs
AbstractMCMC.steps(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler)
```

### Transducer

```@docs
AbstractMCMC.Sample(::AbstractRNG, ::AbstractMCMC.AbstractModel, ::AbstractMCMC.AbstractSampler)
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
```

Two algorithms are provided for parallel sampling with multiple threads and multiple processes, and one allows for the user to sample multiple chains in serial (no parallelization):
```@docs
AbstractMCMC.MCMCThreads
AbstractMCMC.MCMCDistributed
AbstractMCMC.MCMCSerial
```

## Common keyword arguments

Common keyword arguments for regular and parallel sampling (not supported by the iterator and transducer)
are:
- `progress` (default: `AbstractMCMC.PROGRESS[]` which is `true` initially):  toggles progress logging
- `chain_type` (default: `Any`): determines the type of the returned chain
- `callback` (default: `nothing`): if `callback !== nothing`, then
  `callback(rng, model, sampler, sample, iteration)` is called after every sampling step,
  where `sample` is the most recent sample of the Markov chain and `iteration` is the current iteration
- `discard_initial` (default: `0`): number of initial samples that are discarded
- `thinning` (default: `1`): factor by which to thin samples.

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

## Interacting with states of samplers

To make it a bit easier to interact with some arbitrary sampler state, we encourage implementations of `AbstractSampler` to implement the following methods:
```@docs
AbstractMCMC.parameters(state)
AbstractMCMC.setparameters!!(state, parameters)
```
and optionally
```@docs
AbstractMCMC.updatestate!!(state, transition, state_prev)
```
These methods can also be useful for implementing samplers which wraps some inner samplers, e.g. a mixture of samplers.
