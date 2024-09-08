# `state` Interface Functions

We encourage sampler packages to implement the following interface functions for the `state` type(s) they maintain:

```@doc
get_logprob
set_logprob!!
get_params
set_params!!
```

These function will provide a minimum interface to interact with the `state` datatype, which a sampler package doesn't have to expose.

## Using the `state` Interface for block sampling within Gibbs

In this sections, we will demonstrate how a `model` package may use this `state` interface to support a Gibbs sampler that can support blocking sampling using different inference algorithms.

We consider a simple hierarchical model with a normal likelihood, with unknown mean and variance parameters.

```math
\begin{align}
\mu &\sim \text{Normal}(0, 1) \\
\tau^2 &\sim \text{InverseGamma}(1, 1) \\
x_i &\sim \text{Normal}(\mu, \sqrt{\tau^2})
\end{align}
```

We can write the log joint probability function as follows, where for the sake of simplicity for the following steps, we will assume that the `mu` and `tau2` parameters are one-element vectors. And `x` is the data.

```julia
function log_joint(; mu::Vector{Float64}, tau2::Vector{Float64}, x::Vector{Float64})
    # mu is the mean
    # tau2 is the variance
    # x is data

    # μ ~ Normal(0, 1)
    # τ² ~ InverseGamma(1, 1)
    # xᵢ ~ Normal(μ, √τ²)

    logp = 0.0
    mu = only(mu)
    tau2 = only(tau2)

    mu_prior = Normal(0, 1)
    logp += logpdf(mu_prior, mu)

    tau2_prior = InverseGamma(1, 1)
    logp += logpdf(tau2_prior, tau2)

    obs_prior = Normal(mu, sqrt(tau2))
    logp += sum(logpdf(obs_prior, xi) for xi in x)

    return logp
end
```

To make using `LogDensityProblems` interface, we create a simple type for this model.

```julia
abstract type AbstractHierNormal end

struct HierNormal <: AbstractHierNormal
    data::NamedTuple
end

struct ConditionedHierNormal{conditioned_vars} <: AbstractHierNormal
    data::NamedTuple
    conditioned_values::NamedTuple{conditioned_vars}
end
```

where `ConditionedHierNormal` is a type that represents the model conditioned on some variables, and 

```julia
function AbstractPPL.condition(hn::HierNormal, conditioned_values::NamedTuple)
    return ConditionedHierNormal(hn.data, conditioned_values)
end
```

then we can simply write down the `LogDensityProblems` interface for this model.

```julia
function LogDensityProblems.logdensity(
    hn::ConditionedHierNormal{names}, params::AbstractVector
) where {names}
    if Set(names) == Set([:mu]) # conditioned on mu, so params are tau2
        return log_joint(; mu=hn.conditioned_values.mu, tau2=params, x=hn.data.x)
    elseif Set(names) == Set([:tau2]) # conditioned on tau2, so params are mu
        return log_joint(; mu=params, tau2=hn.conditioned_values.tau2, x=hn.data.x)
    else
        error("Unsupported conditioning configuration.")
    end
end

function LogDensityProblems.capabilities(::HierNormal)
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(::ConditionedHierNormal)
    return LogDensityProblems.LogDensityOrder{0}()
end
```

the model should also define a function that allows the recomputation of the log probability given a sampler state.
The reason for this is that, when we break down the joint probability into conditional probabilities, individual conditional probability problems are conditional on the values of the other variables.
Between the Gibbs sampler sweeps, the values of the variables may change, and we need to recompute the log probability of the current state.

A recomputation function could use the `state` interface to return a new state with the updated log probability.
E.g.

```julia
function recompute_logprob!!(hn::ConditionedHierNormal, vals, state)
    return AbstractMCMC.set_logprob!!(state, LogDensityProblems.logdensity(hn, vals))
end
```

where the model doesn't need to know the details of the `state` type, as long as it can access the `log_joint` function.

Additionally, we define a couple of helper functions to transform between the sampler representation and the model representation of the parameters values.
In this simple example, the model representation is a vector, and the sampler representation is a named tuple.

```julia
function flatten(nt::NamedTuple)
    return only(values(nt))
end

function unflatten(vec::AbstractVector, group::Tuple)
    return NamedTuple((only(group) => vec,))
end
```

## Sampler Packages

To illustrate the `AbstractMCMC` interface, we will first implement two very simple Metropolis-Hastings samplers: random walk and static proposal.

Although the interface doesn't force the sampler to implement `Transition` and `State` types, in practice, it has been the convention to do so.

Here we define some bare minimum types to represent the transitions and states.

```julia
struct MHTransition{T}
    params::Vector{T}
end

struct MHState{T}
    params::Vector{T}
    logp::Float64
end
```

Next we define the four `state` interface functions.

```julia
AbstractMCMC.get_params(state::MHState) = state.params
AbstractMCMC.set_params!!(state::MHState, params) = MHState(params, state.logp)
AbstractMCMC.get_logprob(state::MHState) = state.logp
AbstractMCMC.set_logprob!!(state::MHState, logp) = MHState(state.params, logp)
```

These are the functions that was used in the `recompute_logprob!!` function above.

It is very simple to implement the samplers according to the `AbstractMCMC` interface, where we can use `get_logprob` to easily read the log probability of the current state.

```julia
struct RandomWalkMH <: AbstractMCMC.AbstractSampler
    σ::Float64
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::RandomWalkMH,
    args...;
    initial_params,
    kwargs...,
)
    return MHTransition(initial_params),
    MHState(
        initial_params,
        only(LogDensityProblems.logdensity(logdensity_model.logdensity, initial_params)),
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::RandomWalkMH,
    state::MHState,
    args...;
    kwargs...,
)
    params = state.params
    proposal_dist = MvNormal(zeros(length(params)), sampler.σ)
    proposal = params .+ rand(rng, proposal_dist)
    logp_proposal = only(
        LogDensityProblems.logdensity(logdensity_model.logdensity, proposal)
    )

    log_acceptance_ratio = min(0, logp_proposal - get_logprob(state))

    if log(rand(rng)) < log_acceptance_ratio
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, get_logprob(state))
    end
end
```

```julia
struct IndependentMH <: AbstractMCMC.AbstractSampler
    prior_dist::Distribution
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::IndependentMH,
    args...;
    initial_params,
    kwargs...,
)
    return MHTransition(initial_params),
    MHState(
        initial_params,
        only(LogDensityProblems.logdensity(logdensity_model.logdensity, initial_params)),
    )
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::IndependentMH,
    state::MHState,
    args...;
    kwargs...,
)
    params = get_params(state)
    proposal_dist = sampler.prior_dist
    proposal = rand(rng, proposal_dist)
    logp_proposal = only(
        LogDensityProblems.logdensity(logdensity_model.logdensity, proposal)
    )

    log_acceptance_ratio = min(
        0,
        logp_proposal - get_logprob(state) + logpdf(proposal_dist, params) -
        logpdf(proposal_dist, proposal),
    )

    if log(rand(rng)) < log_acceptance_ratio
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, get_logprob(state))
    end
end
```

At last, we can proceed to implement the Gibbs sampler.

```julia
struct Gibbs <: AbstractMCMC.AbstractSampler
    sampler_map::OrderedDict
end

struct GibbsState
    vi::NamedTuple
    states::OrderedDict
end

struct GibbsTransition
    values::NamedTuple
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    spl::Gibbs,
    args...;
    initial_params::NamedTuple,
    kwargs...,
)
    states = OrderedDict()
    for group in keys(spl.sampler_map)
        sub_spl = spl.sampler_map[group]

        vars_to_be_conditioned_on = setdiff(keys(initial_params), group)
        cond_val = NamedTuple{Tuple(vars_to_be_conditioned_on)}(
            Tuple([initial_params[g] for g in vars_to_be_conditioned_on])
        )
        params_val = NamedTuple{Tuple(group)}(Tuple([initial_params[g] for g in group]))
        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(
                    condition(logdensity_model.logdensity, cond_val)
                ),
                sub_spl,
                args...;
                initial_params=flatten(params_val),
                kwargs...,
            ),
        )
        states[group] = sub_state
    end
    return GibbsTransition(initial_params), GibbsState(initial_params, states)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    spl::Gibbs,
    state::GibbsState,
    args...;
    kwargs...,
)
    vi = state.vi
    for group in keys(spl.sampler_map)
        for (group, sub_state) in state.states
            vi = merge(vi, unflatten(get_params(sub_state), group))
        end
        sub_spl = spl.sampler_map[group]
        sub_state = state.states[group]
        group_complement = setdiff(keys(vi), group)
        cond_val = NamedTuple{Tuple(group_complement)}(
            Tuple([vi[g] for g in group_complement])
        )
        cond_logdensity = condition(logdensity_model.logdensity, cond_val)
        sub_state = recompute_logprob!!(cond_logdensity, get_params(sub_state), sub_state)
        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(cond_logdensity),
                sub_spl,
                sub_state,
                args...;
                kwargs...,
            ),
        )
        state.states[group] = sub_state
    end
    for (group, sub_state) in state.states
        vi = merge(vi, unflatten(get_params(sub_state), group))
    end
    return GibbsTransition(vi), GibbsState(vi, state.states)
end
```

Some points worth noting:

1. We are using `OrderedDict` to store the mapping between variables and samplers. The order will determine the order of the Gibbs sweeps.
2. For each conditional probability problem, we need to store the sampler states for each variable group and also the values of all the variables from last iteration.
3. The first step of the Gibbs sampler is to setup the states for each conditional probability problem.
4. In the following steps of the Gibbs sampler, it will do a sweep over all the conditional probability problems, and update the sampler states for each problem. In each step of the sweep, it will do the following:
    - first update the values from the last step of the sweep into the `vi`, which stores the values of all variables at the moment of the Gibbs sweep.
    - condition on the values of all variables that are not in the current group
    - recompute the log probability of the current state, because the values of the variables that are not in the current group may have changed
    - perform a step of the sampler for the conditional probability problem, and update the sampler state
    - update the `vi` with the new values from the sampler state

Again, the `state` interface in AbstractMCMC allows the Gibbs sampler to be agnostic of the details of the sampler state, and acquire the values of the parameters from individual sampler states.

Now we can use the Gibbs sampler to sample from the hierarchical normal model.

First we generate some data,

```julia
N = 100  # Number of data points
mu_true = 0.5  # True mean
tau2_true = 2.0  # True variance

x_data = rand(Normal(mu_true, sqrt(tau2_true)), N)
```

Then we can create a `HierNormal` model, with the data we just generated.

```julia
hn = HierNormal((x=x_data,))
```

Using Gibbs sampling allows us to use random walk MH for `mu` and prior MH for `tau2`, because `tau2` has support only on positive real numbers.

```julia
samples = sample(
    hn,
    Gibbs(
        OrderedDict(
            (:mu,) => RandomWalkMH(1),
            (:tau2,) => IndependentMH(product_distribution([InverseGamma(1, 1)])),
        ),
    ),
    100000;
    initial_params=(mu=[0.0], tau2=[1.0]),
)
```

Then we can extract the samples and compute the mean of the samples.

```julia
mu_samples = [sample.values.mu for sample in samples][20001:end]
tau2_samples = [sample.values.tau2 for sample in samples][20001:end]

mean(mu_samples)
mean(tau2_samples)
```
