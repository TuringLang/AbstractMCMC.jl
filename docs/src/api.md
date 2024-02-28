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
  `callback(rng, model, sampler, sample, state, iteration)` is called after every sampling step,
  where `sample` is the most recent sample of the Markov chain and `state` and `iteration` are the current state and iteration of the sampler
- `discard_initial` (default: `0`): number of initial samples that are discarded
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

## Interacting with states of samplers

To make it a bit easier to interact with some arbitrary sampler state, we encourage implementations of `AbstractSampler` to implement the following methods:
```@docs
AbstractMCMC.realize
AbstractMCMC.realize!!
```
and optionally
```@docs
AbstractMCMC.updatestate!!(state, transition, state_prev)
```
These methods can also be useful for implementing samplers which wraps some inner samplers, e.g. a mixture of samplers.

### Example: `MixtureSampler`

In a `MixtureSampler` we need two things:
- `components`: collection of samplers.
- `weights`: collection of weights representing the probability of chosing the corresponding sampler.

```julia
struct MixtureSampler{W,C} <: AbstractMCMC.AbstractSampler
    components::C
    weights::W
end
```

To implement the state, we need to keep track of a couple of things:
- `index`: the index of the sampler used in this `step`.
- `transition`: the transition resulting from this `step`.
- `states`: the current states of _all_ the components.
Two aspects of this might seem a bit strange:
1. We need to keep track of the states of _all_ components rather than just the state for the sampler we used previously.
2. We need to put the `transition` from the `step` into the state.

The reason for (1) is that lots of samplers keep track of more than just the previous realizations of the variables, e.g. in `AdvancedHMC.jl` we keep track of the momentum used, the metric used, etc.

For (2) the reason is similar: some samplers might keep track of the variables _in the state_ differently, e.g. you might have a sampler which is _independent_ of the current realizations and the state is simply `nothing`. 

Hence, we need the `transition`, which should always contain the realizations, to make sure we can resume from the same point in the space in the next `step`.
```julia
struct MixtureState{T,S}
    index::Int
    transition::T
    states::S
end
```
The `step` for a `MixtureSampler` is defined by the following generative process
```math
\begin{aligned}
i &\sim \mathrm{Categorical}(w_1, \dots, w_k) \\
X_t &\sim \mathcal{K}_i(\cdot \mid X_{t - 1})
\end{aligned}
```
where ``\mathcal{K}_i`` denotes the i-th kernel/sampler, and ``w_i`` denotes the weight/probability of choosing the i-th sampler.
[`AbstractMCMC.updatestate!!`](@ref) comes into play in defining/computing ``\mathcal{K}_i(\cdot \mid X_{t - 1})`` since ``X_{t - 1}`` could be coming from a different sampler. 

If we let `state` be the current `MixtureState`, `i` the current component, and `i_prev` is the previous component we sampled from, then this translates into the following piece of code:

```julia
# Update the corresponding state, i.e. `state.states[i]`, using
# the state and transition from the previous iteration.
state_current = AbstractMCMC.updatestate!!(
    state.states[i], state.states[i_prev], state.transition
)

# Take a `step` for this sampler using the updated state.
transition, state_current = AbstractMCMC.step(
    rng, model, sampler_current, sampler_state;
    kwargs...
)
```

The full [`AbstractMCMC.step`](@ref) implementation would then be something like:

```julia
function AbstractMCMC.step(rng, model::AbstractMCMC.AbstractModel, sampler::MixtureSampler, state; kwargs...)
    # Sample the component to use in this `step`.
    i = rand(Categorical(sampler.weights))
    sampler_current = sampler.components[i]

    # Update the corresponding state, i.e. `state.states[i]`, using
    # the state and transition from the previous iteration.
    i_prev = state.index
    state_current = AbstractMCMC.updatestate!!(
        model, state.states[i], state.states[i_prev], state.transition
    )

    # Take a `step` for this sampler using the updated state.
    transition, state_current = AbstractMCMC.step(
        rng, model, sampler_current, state_current;
        kwargs...
    )

    # Create the new states.
    # NOTE: Code below will result in `states_new` being a `Vector`.
    # If we wanted to allow usage of alternative containers, e.g. `Tuple`,
    # it would be better to use something like `@set states[i] = state_current`
    # where `@set` is from Setfield.jl.
    states_new = map(1:length(state.states)) do j
        if j == i
            # Replace the i-th state with the new one.
            state_current
        else
            # Otherwise we just carry over the previous ones.
            state.states[j]
        end
    end

    # Create the new `MixtureState`.
    state_new = MixtureState(i, transition, states_new)

    return transition, state_new
end
```

And for the initial [`AbstractMCMC.step`](@ref) we have:

```julia
function AbstractMCMC.step(rng, model::AbstractMCMC.AbstractModel, sampler::MixtureSampler; kwargs...)
    # Initialize every state.
    transitions_and_states = map(sampler.components) do spl
        AbstractMCMC.step(rng, model, spl; kwargs...)
    end

    # Sample the component to use this `step`.
    i = rand(Categorical(sampler.weights))
    # Extract the corresponding transition.
    transition = first(transitions_and_states[i])
    # Extract states.
    states = map(last, transitions_and_states)
    # Create new `MixtureState`.
    state = MixtureState(i, transition, states)

    return transition, state
end
```

Suppose we then wanted to use this with some of the packages which implements AbstractMCMC.jl's interface, e.g. [`AdvancedMH.jl`](https://github.com/TuringLang/AdvancedMH.jl), then we'd simply have to implement `realize` and `realize!!`:

```julia
function AbstractMCMC.updatestate!!(model, ::AdvancedMH.Transition, state_prev::AdvancedMH.Transition)
    # Let's `deepcopy` just to be certain.
    return deepcopy(state_prev)
end
```

To use `MixtureSampler` with two samplers `sampler1` and `sampler2` from `AdvancedMH.jl` as components, we'd simply do

```julia
sampler = MixtureSampler([sampler1, sampler2], [0.1, 0.9])
transition, state = AbstractMCMC.step(rng, model, sampler)
while ...
    transition, state = AbstractMCMC.step(rng, model, sampler, state)
end
```

As a final note, there is one potential issue we haven't really addressed in the above implementation: a lot of samplers have their own implementations of `AbstractMCMC.AbstractModel` which means that we would also have to ensure that all the different samplers we are using would be compatible with the same model. A very easy way to fix this would be to just add a struct called `ManyModels` supporting `getindex`, e.g. `models[i]` would return the i-th `model`:

```julia
struct ManyModels{M} <: AbstractMCMC.AbstractModel
    models::M
end

Base.getindex(model::ManyModels, I...) = model.models[I...]
```

Then the above `step` would just extract the `model` corresponding to the current sampler:

```julia
# Take a `step` for this sampler using the updated state.
transition, state_current = AbstractMCMC.step(
    rng, model[i], sampler_current, state_current;
    kwargs...
)
```

This issue should eventually disappear as the community moves towards a unified approach to implement `AbstractMCMC.AbstractModel`.
