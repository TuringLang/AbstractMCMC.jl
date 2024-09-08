using Distributions

struct MHState{T}
    params::Vector{T}
    logp::Float64
end

struct MHTransition{T}
    params::Vector{T}
end

function AbstractMCMC.LogDensityProblems.logdensity(logdensity_function, state::MHState)
    # recompute the logdensity, instead of using the one stored in the state
    return AbstractMCMC.LogDensityProblems.logdensity(logdensity_function, state.params)
end

function Base.vec(state::MHState)
    return state.params
end

struct RandomWalkMH <: AbstractMCMC.AbstractSampler
    σ::Float64
end

struct IndependentMH <: AbstractMCMC.AbstractSampler
    proposal_dist::Distributions.Distribution
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Union{RandomWalkMH,IndependentMH},
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
    sampler::Union{RandomWalkMH,IndependentMH},
    state::MHState,
    args...;
    kwargs...,
)
    params = state.params
    proposal_dist =
        sampler isa RandomWalkMH ? MvNormal(state.params, sampler.σ) : sampler.proposal_dist
    proposal = rand(rng, proposal_dist)
    logp_proposal = only(
        LogDensityProblems.logdensity(logdensity_model.logdensity, proposal)
    )

    if log(rand(rng)) <
        compute_log_acceptance_ratio(sampler, state, proposal, logp_proposal)
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, state.logp)
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
