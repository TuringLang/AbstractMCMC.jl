using AbstractMCMC: AbstractMCMC, LogDensityProblems
using Distributions
using Random

abstract type AbstractMHSampler <: AbstractMCMC.AbstractSampler end

struct MHState{T}
    params::Vector{T}
    logp::Float64
end

# Interface 3: (state::MHState)(logp::Float64)
# This function allows the state to be updated with a new log probability.
# ! this makes state into a Julia functor
(state::MHState)(logp::Float64) = MHState(state.params, logp)

struct MHTransition{T}
    params::Vector{T}
end

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
