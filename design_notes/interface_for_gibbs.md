# Notes on Potential Future Interface for Gibbs Sampling Support

## Background

This document was written after [PR #144](https://github.com/TuringLang/AbstractMCMC.jl/pull/144) was closed.

It was last updated on October 15, 2024. At that time:

- _AbstractMCMC.jl_ was on version 5.5.0  
- _Turing.jl_ was on version 0.34.1

The goal is to document some of the considerations that went into the closed PR mentioned above.

## Gibbs Sampling Considerations

### Recomputing Log Densities for Parameter Groups

Let's consider splitting the model parameters into several groups (assuming the grouping stays fixed between iterations). Each parameter group will have a corresponding sampler state (along with the sampler used for that group).

In the general case, the log densities stored in the states will be incorrect at the time of sampling each group. This is because the values of the other two parameter groups can change from when the current log density was computed, as they get updated within the Gibbs sweep.

### Current Approach: `recompute_logp!!`

_Turing.jl_'s current solution, at the time of writing this, is the `recompute_logp!!` function (see [Tor's comment](https://github.com/TuringLang/AbstractMCMC.jl/issues/85#issuecomment-2061300622) and the [`Gibbs` PR](https://github.com/TuringLang/Turing.jl/pull/2099)).

Here's an example implementation of this function for _AbstractHMC.jl_ ([permalink](https://github.com/TuringLang/Turing.jl/blob/24e68701b01695bffe69eda9e948e910c1ae2996/src/mcmc/abstractmcmc.jl#L77C1-L90C1)):

```julia
function recompute_logprob!!(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.LogDensityModel,
    sampler::AdvancedHMC.AbstractHMCSampler,
    state::AdvancedHMC.HMCState,
)
    # Construct hamiltionian.
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, model)
    # Re-compute the log-probability and gradient.
    return Accessors.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, state.transition.z.θ, state.transition.z.r
    )
end
```

### Alternative Approach Proposed in [PR #144](https://github.com/TuringLang/AbstractMCMC.jl/pull/144)

The proposal is to separate `recompute_logp!!` into two functions:

1. A function to compute the log density given the model and sampler state
2. A function to set the computed log density in the sampler state

There are a few considerations with this approach:

- Computing the log density involves a `model`, which may not be defined by the sampler package in the general case. It's unclear if this interface is appropriate, as the model details might be needed to calculate the log density. However, in many situations, the `LogDensityProblems` interface (`LogDensityModel` in `AbstractMCMC`) could be sufficient.
  - One interfacial consideration is that `LogDensityProblems.logdensity` expects a vector input. For our use case, we may want to reuse the log density stored in the state instead of recomputing it each time. This would require modifying `logdensity` to accept a sampler state and potentially a boolean flag to indicate whether to recompute the log density or not.
- In some cases, samplers require more than just the log joint density. They may also need the log likelihood and log prior separately (see [this discussion](https://github.com/TuringLang/AbstractMCMC.jl/issues/112)).

## Potential Path Forward

A reasonable next step would be to explore an interface similar to `LogDensityProblems.logdensity`, but with the ability to compute both the log prior and log likelihood. It should also accept alternative inputs and keyword arguments.

To complement this computation interface, we would need functions to `get` and `set` the log likelihood and log prior from/to the sampler states.

For situations where model-specific details are required to compute the log density from a sampler state, the necessary abstractions are not yet clear. We will need to consider appropriate abstractions as such use cases emerge.

## Additional Notes on a More Independent Gibbs Implementation

### Regarding `AbstractPPL.condition`

While the `condition` function is a promising idea for Gibbs sampling, it is not currently being utilized in _Turing.jl_'s implementation. Instead, _Turing.jl_ uses a `GibbsContext` for reasons outlined [here](https://github.com/TuringLang/Turing.jl/blob/3c91eec43176d26048b810aae0f6f2fac0686cfa/src/experimental/gibbs.jl#L1-L12). Additionally, _JuliaBUGS_ requires caching the Markov blanket when calling `condition`, which means the proposed `Gibbs` implementation in this PR would not be fully compatible.

### Samplers Should Not Manage Variable Names

To make `AbstractMCMC.Gibbs` more independent and flexible, it should manage a mapping of `range → sampler` rather than `variable name → sampler`. This means it would maintain a vector of parameter values internally. The responsibility of managing both the variable names and any necessary transformations should be handled by a higher-level interface such as `AbstractPPL` or `DynamicPPL`.

By separating these concerns, `AbstractMCMC.Gibbs` can focus on the core Gibbs sampling logic while the PPL interface handles the specifics of variable naming and transformations. This modular approach allows for greater flexibility and easier integration with different PPL frameworks.

However, the issue arises when we have transdimensional parameters. In such cases, the parameter space can change during sampling, making it challenging to maintain a fixed mapping between ranges and samplers.
