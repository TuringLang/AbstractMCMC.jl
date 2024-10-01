# On `AbstractMCMC` Interface Supporting `Gibbs`

This is written at Oct 1st, 2024. Version of packages described in this passage are:

* `Turing.jl`: 0.34.1

In this passage, `Gibbs` refers to `Experimental.Gibbs`.

## Current Implementation of `Gibbs` in `Turing`

Here I describe the current implementation of `Gibbs` in `Turing` and the interface it requires from its sampler states.

### Interface 1: `getparams`

From the [definition of `GibbsState`](https://github.com/TuringLang/Turing.jl/blob/3c91eec43176d26048b810aae0f6f2fac0686cfa/src/experimental/gibbs.jl#L244-L248), we can see that a `vi::DynamicPPL.AbstractVarInfo` field is used to keep track of the names and values of parameters and the log density. The `states` field collects the sampler-specific *state*s.

(The *link*ing of *varinfo*s is omitted in this discussion.)
A local `VarInfo` is initially created with `DynamicPPL.subset(::VarInfo, ::Vector{<:VarName})` to make the conditioned model. After the Gibbs step, an updated `varinfo` is obtained by calling `Turing.Inference.varinfo` on the sampler state.

For samplers and their states defined in `Turing` (including `DynamicHMC`, as `DynamicNUTSState` is defined by `Turing` in the package extension), we (à la `Turing.jl` package) assume that the *state*s all have a field called `vi`. Then `varinfo(_some_sampler_state_)` is simply `varinfo(state) = state.vi` (defined in [`src/mcmc/gibbs.jl`](https://github.com/TuringLang/Turing.jl/blob/3c91eec43176d26048b810aae0f6f2fac0686cfa/src/mcmc/gibbs.jl#L97)). (`GibbsState` conforms to this assumption.)

For `ExternalSamplers`, we currently only support `AdvancedHMC` and `AdvancedMH`. The mechanism is as follows: at the end of the `step` call with an external sampler, [`transition_to_turing` and `state_to_turing` are called](https://github.com/TuringLang/Turing.jl/blob/3c91eec43176d26048b810aae0f6f2fac0686cfa/src/mcmc/abstractmcmc.jl#L147). These two functions then call `getparams` on the sampler state of the external samplers. `getparams` for `AdvancedHMC.HMCState` and `AdvancedMH.Transition` (`AdvancedMH` uses `Transition` as state) are defined in `abstractmcmc.jl`.

Thus, the first interface emerges: `getparams`. As `getparams` is designed to be implemented by a sampler that works with the `LogDensityProblems` interface, it makes sense for `getparams` to return a vector of `Real`s. The `logdensity_problem` should then be responsible for performing the transformation between its underlying representation and the vector of `Real`s.

It's worth noting that:

* `getparams` is not a function specific for `Gibbs`. It is required for the current support of external samplers.
* There is another [`getparams`](https://github.com/TuringLang/Turing.jl/blob/3c91eec43176d26048b810aae0f6f2fac0686cfa/src/mcmc/Inference.jl#L328-L351) in `Turing.jl` that takes *model* and *varinfo*, then returns a `NamedTuple`.

### Interface 2: `recompute_logp!!`

Consider a model with multiple groups of variables, say $\theta_1, \theta_2, \ldots, \theta_k$. At the beginning of the $t$-th Gibbs step, the model parameters in the `GibbsState` are typically updated and different from the $(t-1)$-th step. The `GibbsState` maintains $k$ sub-states, one for each variable group, denoted as $\text{state}_{t,1}, \text{state}_{t,2}, \ldots, \text{state}_{t,k}$.

The parameter values in each sub-state, i.e., $\theta_{t,i}$ in $\text{state}_{t,i}$, are always in sync with the corresponding values in the `GibbsState`. At the end of the $t$-th Gibbs step, $\text{state}_{t,i}$ will store the log density of the $i$-th variable group conditioned on all other variable groups at their values from step $t$, denoted as $\log p(\theta_{t,i} \mid \theta_{t,-i})$. This log density is equal to the joint log density of the whole model evaluated at the current parameter values $(\theta_{t,1}, \ldots, \theta_{t,k})$.

However, the log density stored in each sub-state is in general not equal to the log density needed for the next Gibbs step at $t+1$, i.e., $\log p(\theta_{t,i} \mid \theta_{t+1,-i})$. This is because the values of the other variable groups $\theta_{-i}$ will have been updated in the Gibbs step from $t$ to $t+1$, changing the conditioning set. Therefore, the log density typically needs to be recomputed at each Gibbs step to account for the updated values of the conditioning variables.

Only in certain special cases, the recomputation can be skipped. For example, in a Metropolis-Hastings step where the proposal is rejected for all other variable groups, i.e., $\theta_{t+1,-i} = \theta_{t,-i}$, the log density $\log p(\theta_{t,i} \mid \theta_{t,-i})$ remains valid and doesn't need to be recomputed.

The `recompute_logp!!` function in `abstractmcmc.jl` handles this recomputation. It takes an updated conditioned log density function $\log p(\cdot \mid \theta_{t+1,j})$ and the parameter values $\theta_{t,i}$ stored in $\text{state}_{t,i}$ to compute the updated log density $\log p(\theta_{t,i} \mid \theta_{t+1,j})$.

## Proposed Interface

The two functions `getparams` and `recompute_logp!!` form a minimal interface to support the `Gibbs` implementation. However, there are concerns about introducing them directly into `AbstractMCMC`. The main reason is that `AbstractMCMC` is a root dependency of the `Turing` packages, so we want to be very careful with new releases.

Here, I propose some alternative functions that achieve the same functionality as `getparams` and `recompute_logp!!`, but without introducing new interface functions.

For `getparams`, I propose we use `Base.vec`. It is a `Base` function, so there's no need to export anything from `AbstractMCMC`. Since `getparams` should return a vector, using `vec` makes sense. The concern is that, officially, `Base.vec` is defined for `AbstractArray`, so it remains a question whether we should only introduce `vec` in the absence of other `AbstractArray` interfaces.

For `recompute_logp!!`, I propose we overload `LogDensityProblems.logdensity(logdensity_model::AbstractMCMC.LogDensityModel, state::State; recompute_logp=true)` to compute the log probability. If `recompute_logp` is `true`, it should recompute the log probability of the state. Otherwise, it could use the log probability stored in the state. To allow updating the log probability stored in the state, samplers should define outer constructor for their state types `StateType(state::StateType, logp)` that takes an existing `state` and a log probability value `logp`, and returns a new state of the same type with the updated log probability.

While overloading `LogDensityProblems.logdensity` to take a state object instead of a vector for the second argument somewhat deviate from the interface in `LogDensityProblems`, I believe it provides a clean and extensible solution for handling log probability recomputation within the existing interface.

An example demonstrating these interfaces is provided in `src/state_interface.md`.

## A More Standalone `Gibbs` Implementation

`Gibbs` should not manage a `variable name → sampler` but rather `range → sampler`, i.e. it maintain a vector of parameter values. while `logdensity_problem` should manage both the name and transformations.
