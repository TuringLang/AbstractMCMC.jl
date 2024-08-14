struct RWMH <: AbstractMCMC.AbstractSampler
    σ
end

struct MHTransition{T} where {T}
    params::T
end

struct MHState{T} where {T}
    params::T
    logp::Float64
end

getparams(state::MHState) = state.params
setparams!!(state::MHState, params) = MHState(params, state.logp)
getlogp(state::MHState) = state.logp
setlogp!!(state::MHState, logp) = MHState(state.params, logp)

function AbstractMCMC.step(rng::AbstractRNG, logdensity, sampler::RWMH, args...; kwargs...)
    params = rand(rng, LogDensityProblems.dimension(logdensity))
    return MHTransition(params),
    MHState(params, LogDensityProblems.logdensity(logdensity, params))
end

function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::RWMH, state::MHState, args...; kwargs...
)
    params = getparams(state)
    proposal_dist = MvNormal(params, sampler.σ)
    proposal = rand(rng, proposal_dist)
    logp_proposal = logpdf(proposal_dist, proposal)
    accepted = log(rand(rng)) < log1pexp(min(0, logp_proposal - getlogp(state)))
    if accepted
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, getlogp(state))
    end
end

struct PriorMH <: AbstractMCMC.AbstractSampler
    prior_dist
end

function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::PriorMH, args...; kwargs...
)
    params = rand(rng, sampler.prior_dist)
    return MHTransition(params), MHState(params, logdensity(params))
end

function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::PriorMH, state::MHState, args...; kwargs...
)
    params = getparams(state)
    proposal_dist = sampler.prior_dist
    proposal = rand(rng, proposal_dist)
    logp_proposal = logpdf(proposal_dist, proposal)
    accepted = log(rand(rng)) < log1pexp(min(0, logp_proposal - getlogp(state)))
    if accepted
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, getlogp(state))
    end
end
