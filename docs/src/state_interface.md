# Interface For Sampler `state` and Gibbs Sampling

We encourage sampler packages to implement the following interface functions for the `state` type(s) they maintain:

```julia
LogDensityProblems.logdensity(logdensity_model::AbstractMCMC.LogDensityModel, state::MHState; recompute_logp=true)
```

This function takes the logdensity model and the state, and returns the log probability of the state.
If `recompute_logp` is `true`, it should recompute the log probability of the state.
Otherwise, it could use the log probability stored in the state.

```julia
Base.vec(state)
```

This function takes the state and returns a vector of the parameter values stored in the state.

```julia
(state::StateType)(logp::Float64)
```

This function takes the state and a log probability value, and returns a new state with the updated log probability.

These functions provide a minimal interface to interact with the `state` datatype, which a sampler package can optionally implement.
The interface facilitates the implementation of "meta-algorithms" that combine different samplers.
We will demonstrate how it can be used to implement Gibbs sampling in the following sections.

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

struct HierNormal{Tdata<:NamedTuple} <: AbstractHierNormal
    data::Tdata
end

struct ConditionedHierNormal{Tdata<:NamedTuple,Tconditioned_vars<:NamedTuple} <:
       AbstractHierNormal
    data::Tdata

    " The variable to be conditioned on and its value"
    conditioned_values::Tconditioned_vars
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
    hier_normal_model::ConditionedHierNormal{Tdata,Tconditioned_vars},
    params::AbstractVector,
) where {Tdata,Tconditioned_vars}
    variable_to_condition = only(fieldnames(Tconditioned_vars))
    data = hier_normal_model.data
    conditioned_values = hier_normal_model.conditioned_values

    if variable_to_condition == :mu
        return log_joint(; mu=conditioned_values.mu, tau2=params, x=data.x)
    elseif variable_to_condition == :tau2
        return log_joint(; mu=params, tau2=conditioned_values.tau2, x=data.x)
    else
        error("Unsupported conditioning variable: $variable_to_condition")
    end
end

function LogDensityProblems.capabilities(::HierNormal)
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(::ConditionedHierNormal)
    return LogDensityProblems.LogDensityOrder{0}()
end
```

### Implementing A Sampler with `AbstractMCMC` Interface

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

Next we define the  `state` interface functions mentioned at the beginning of this section.

```julia
# Interface 1: LogDensityProblems.logdensity
# This function takes the logdensity function and the state (state is defined by the sampler package)
# and returns the logdensity. It allows for optional recomputation of the log probability.
# If recomputation is not needed, it returns the stored log probability from the state.
function LogDensityProblems.logdensity(
    logdensity_model::AbstractMCMC.LogDensityModel, state::MHState; recompute_logp=true
)
    logdensity_function = logdensity_model.logdensity
    return if recompute_logp
        AbstractMCMC.LogDensityProblems.logdensity(logdensity_function, state.params)
    else
        state.logp
    end
end

# Interface 2: Base.vec
# This function takes a state and returns a vector of the parameter values stored in the state.
# It is part of the interface for interacting with the state object.
Base.vec(state::MHState) = state.params

# Interface 3: (state::MHState)(logp::Float64)
# This function allows the state to be updated with a new log probability.
# ! this makes state into a Julia functor
(state::MHState)(logp::Float64) = MHState(state.params, logp)
```

It is very simple to implement the samplers according to the `AbstractMCMC` interface, where we can use `LogDensityProblems.logdensity` to easily read the log probability of the current state.

```julia
"""
    RandomWalkMH{T} <: AbstractMCMC.AbstractSampler

A random walk Metropolis-Hastings sampler with a normal proposal distribution. The field σ
is the standard deviation of the proposal distribution.
"""
struct RandomWalkMH{T} <: AbstractMHSampler
    σ::T
end

"""
    IndependentMH{T} <: AbstractMCMC.AbstractSampler

A Metropolis-Hastings sampler with an independent proposal distribution.
"""
struct IndependentMH{T} <: AbstractMHSampler
    proposal_dist::T
end

# the first step of the sampler
function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMHSampler,
    args...;
    initial_params,
    kwargs...,
)
    logdensity_function = logdensity_model.logdensity
    transition = MHTransition(initial_params)
    state = MHState(
        initial_params,
        only(LogDensityProblems.logdensity(logdensity_function, initial_params)),
    )

    return transition, state
end

@inline get_proposal_dist(sampler::RandomWalkMH, current_params::Vector{Float64}) =
    MvNormal(current_params, sampler.σ)
@inline get_proposal_dist(sampler::IndependentMH, current_params::Vector{T}) where {T} =
    sampler.proposal_dist

# the subsequent steps of the sampler
function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::AbstractMHSampler,
    state::MHState,
    args...;
    kwargs...,
)
    logdensity_function = logdensity_model.logdensity
    current_params = state.params
    proposal_dist = get_proposal_dist(sampler, current_params)
    proposed_params = rand(rng, proposal_dist)
    logp_proposal = only(
        LogDensityProblems.logdensity(logdensity_function, proposed_params)
    )

    if log(rand(rng)) <
        compute_log_acceptance_ratio(sampler, state, proposed_params, logp_proposal)
        return MHTransition(proposed_params), MHState(proposed_params, logp_proposal)
    else
        return MHTransition(current_params), MHState(current_params, state.logp)
    end
end

function compute_log_acceptance_ratio(
    ::RandomWalkMH, state::MHState, ::Vector{Float64}, logp_proposal::Float64
)
    return min(0, logp_proposal - state.logp)
end

function compute_log_acceptance_ratio(
    sampler::IndependentMH, state::MHState, proposal::Vector{T}, logp_proposal::Float64
) where {T}
    return min(
        0,
        logp_proposal - state.logp + logpdf(sampler.proposal_dist, state.params) -
        logpdf(sampler.proposal_dist, proposal),
    )
end
```

At last, we can proceed to implement a very simple Gibbs sampler.

```julia
struct Gibbs{T<:NamedTuple} <: AbstractMCMC.AbstractSampler
    "Maps variables to their samplers."
    sampler_map::T
end

struct GibbsState{TraceNT<:NamedTuple,StateNT<:NamedTuple,SizeNT<:NamedTuple}
    "Contains the values of all parameters up to the last iteration."
    trace::TraceNT

    "Maps parameters to their sampler-specific MCMC states."
    mcmc_states::StateNT

    "Maps parameters to their sizes."
    variable_sizes::SizeNT
end

struct GibbsTransition{ValuesNT<:NamedTuple}
    "Realizations of the parameters, this is considered a \"sample\" in the MCMC chain."
    values::ValuesNT
end

"""
    update_trace(trace::NamedTuple, gibbs_state::GibbsState)

Update the trace with the values from the MCMC states of the sub-problems.
"""
function update_trace(
    trace::NamedTuple{trace_names}, gibbs_state::GibbsState{TraceNT,StateNT,SizeNT}
) where {trace_names,TraceNT,StateNT,SizeNT}
    for parameter_variable in fieldnames(StateNT)
        sub_state = gibbs_state.mcmc_states[parameter_variable]
        sub_state_params_values = Base.vec(sub_state)
        reshaped_sub_state_params_values = reshape(
            sub_state_params_values, gibbs_state.variable_sizes[parameter_variable]
        )
        unflattened_sub_state_params = NamedTuple{(parameter_variable,)}((
            reshaped_sub_state_params_values,
        ))
        trace = merge(trace, unflattened_sub_state_params)
    end
    return trace
end

function error_if_not_fully_initialized(
    initial_params::NamedTuple{ParamNames}, sampler::Gibbs{<:NamedTuple{SamplerNames}}
) where {ParamNames,SamplerNames}
    if Set(ParamNames) != Set(SamplerNames)
        throw(
            ArgumentError(
                "initial_params must contain all parameters in the model, expected $(SamplerNames), got $(ParamNames)",
            ),
        )
    end
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs{Tsamplingmap};
    initial_params::NamedTuple,
    kwargs...,
) where {Tsamplingmap}
    error_if_not_fully_initialized(initial_params, sampler)

    model_parameter_names = fieldnames(Tsamplingmap)
    results = map(model_parameter_names) do parameter_variable
        sub_sampler = sampler.sampler_map[parameter_variable]

        variables_to_be_conditioned_on = setdiff(
            model_parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([initial_params[g] for g in variables_to_be_conditioned_on])
        )

        # LogDensityProblems' `logdensity` function expects a single vector of real numbers
        # `Gibbs` stores the parameters as a named tuple, thus we need to flatten the sub_problem_parameters_values
        # and unflatten after the sampling step
        flattened_sub_problem_parameters_values = vec(initial_params[parameter_variable])

        sub_logdensity_model = AbstractMCMC.LogDensityModel(
            AbstractPPL.condition(
                logdensity_model.logdensity, conditioning_variables_values
            ),
        )
        sub_state = last(
            AbstractMCMC.step(
                rng,
                sub_logdensity_model,
                sub_sampler;
                initial_params=flattened_sub_problem_parameters_values,
                kwargs...,
            ),
        )
        (sub_state, size(initial_params[parameter_variable]))
    end

    mcmc_states_tuple = first.(results)
    variable_sizes_tuple = last.(results)

    gibbs_state = GibbsState(
        initial_params,
        NamedTuple{Tuple(model_parameter_names)}(mcmc_states_tuple),
        NamedTuple{Tuple(model_parameter_names)}(variable_sizes_tuple),
    )

    trace = update_trace(NamedTuple(), gibbs_state)
    return GibbsTransition(trace), gibbs_state
end

# subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs{Tsamplingmap},
    gibbs_state::GibbsState;
    kwargs...,
) where {Tsamplingmap}
    trace = gibbs_state.trace
    mcmc_states = gibbs_state.mcmc_states
    variable_sizes = gibbs_state.variable_sizes

    model_parameter_names = fieldnames(Tsamplingmap)
    mcmc_states = map(model_parameter_names) do parameter_variable
        sub_sampler = sampler.sampler_map[parameter_variable]
        sub_state = mcmc_states[parameter_variable]
        variables_to_be_conditioned_on = setdiff(
            model_parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([trace[g] for g in variables_to_be_conditioned_on])
        )
        cond_logdensity = AbstractPPL.condition(
            logdensity_model.logdensity, conditioning_variables_values
        )
        cond_logdensity_model = AbstractMCMC.LogDensityModel(cond_logdensity)

        logp = LogDensityProblems.logdensity(cond_logdensity_model, sub_state)
        sub_state = (sub_state)(logp)
        sub_state = last(
            AbstractMCMC.step(
                rng, cond_logdensity_model, sub_sampler, sub_state; kwargs...
            ),
        )
        trace = update_trace(trace, gibbs_state)
        sub_state
    end
    mcmc_states = NamedTuple{Tuple(model_parameter_names)}(mcmc_states)

    return GibbsTransition(trace), GibbsState(trace, mcmc_states, variable_sizes)
end
```

We are using `NamedTuple` to store the mapping between variables and samplers. The order will determine the order of the Gibbs sweeps. A limitation is that exactly one sampler for each variable is required, which means it is less flexible than Gibbs in `Turing.jl`.

We uses the `AbstractPPL.condition` to devide the full model into smaller conditional probability problems.
And each conditional probability problem corresponds to a sampler and corresponding state.

The `Gibbs` sampler has the same interface as other samplers in `AbstractMCMC` (we don't implement the above state interface for `GibbsState` to keep it simple, but it can be implemented similarly).

The Gibbs sampler operates in two main phases:

1. Initialization:
   - Set up initial states for each conditional probability problem.

2. Iterative Sampling:
   For each iteration, the sampler performs a sweep over all conditional probability problems:

   a. Condition on other variables:
      - Fix the values of all variables except the current one.
   b. Update current variable:
      - Recompute the log probability of the current state, as other variables may have changed:
        - Use `LogDensityProblems.logdensity(cond_logdensity_model, sub_state)` to get the new log probability.
        - Update the state with `sub_state = sub_state(logp)` to incorporate the new log probability.
      - Perform a sampling step for the current conditional probability problem:
        - Use `AbstractMCMC.step(rng, cond_logdensity_model, sub_sampler, sub_state; kwargs...)` to generate a new state.
      - Update the global trace:
        - Extract parameter values from the new state using `Base.vec(new_sub_state)`.
        - Incorporate these values into the overall Gibbs state trace.

This process allows the Gibbs sampler to iteratively update each variable while conditioning on the others, gradually exploring the joint distribution of all variables.

Now we can use the Gibbs sampler to sample from the hierarchical Normal model.

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
    Gibbs((
        mu=RandomWalkMH(0.3),
        tau2=IndependentMH(product_distribution([InverseGamma(1, 1)])),
    )),
    10000;
    initial_params=(mu=[0.0], tau2=[1.0]),
)
```

Then we can extract the samples and compute the mean of the samples.

```julia
mu_samples = [sample.values.mu for sample in samples][20001:end]
tau2_samples = [sample.values.tau2 for sample in samples][20001:end]

mean(mu_samples)
mean(tau2_samples)
(mu_mean, tau2_mean)
```

the result should looks like:

```julia
(4.995812149309413, 1.9372372289677886)
```

which is close to the true values `(5, 2)`.
