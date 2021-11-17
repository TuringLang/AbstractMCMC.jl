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
AbstractMCMC.parameters
AbstractMCMC.setparameters!!
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
For (2) the reason is similar: some samplers might keep track of the variables _in the state_ differently, e.g. maybe the sampler is working in a transformed space but returns the samples in the original space, or maybe the sampler is even independent from the current realizations and the state is simply `nothing`. Hence, we need the `transition`, which should always contain the realizations, to make sure we can resume from the same point in the space in the next `step`.
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
[`AbstractMCMC.updatestate!!`](@ref) comes into play in defining/computing ``\mathcal{K}_i(\cdot \mid X_{t - 1})`` since ``X_{t - 1}`` could be coming from a different sampler. If we let `state` be the current `MixtureState`, `i` the current component, and `i_prev` is the previous component we sampled from, then this translates into the following piece of code:

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
        state.states[i], state.states[i_prev], state.transition
    )

    # Take a `step` for this sampler using the updated state.
    transition, state_current = AbstractMCMC.step(
        rng, model, sampler_current, state_current;
        kwargs...
    )

    # Create the new states.
    # NOTE: A better approach would be to use `Setfield.@set state.states[i] = ...`
    # but to keep this demo self-contained, we don't.
    states_new = ntuple(1:length(state.states)) do j
        if j != i
            state.states[i]
        else
            state_inner
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
    transition = first(transition_and_states[i])
    # Extract states.
    states = map(last, transitions_and_states)
    # Create new `MixtureState`.
    state = MixtureState(i, transition, states)

    return transition, state
end
```

To use `MixtureSampler` with two samplers `sampler1` and `sampler2` as components, we'd simply do

```julia
sampler = MixtureSampler((0.1, 0.9), (sampler1, sampler2))
transition, state = AbstractMCMC.step(rng, model, sampler)
while ...
    transition, state = AbstractMCMC.step(rng, model, sampler, state)
end
```

As a final note, there is one potential issue we haven't really addressed in the above implementation: a lot of samplers have their own implementations of `AbstractMCMC.AbstractModel` which means that we would also have to ensure that all the different samplers we are using would be compatible with the same model. A very easy way to fix this would be to just add a struct called `ManyModels` supporting `getindex`, e.g. `models[i]` would return the i-th `model`, and then the above `step` would just extract the `model` corresponding to the current sampler. This issue should eventually disappear as the community moves towards a unified approach to implement `AbstractMCMC.AbstractModel`.
