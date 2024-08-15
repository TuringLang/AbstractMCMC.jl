struct MHTransition{T}
    params::Vector{T}
end

struct MHState{T}
    params::Vector{T}
    logp::Float64
end

getparams(state::MHState) = state.params
setparams!!(state::MHState, params) = MHState(params, state.logp)
getlogp(state::MHState) = state.logp
setlogp!!(state::MHState, logp) = MHState(state.params, logp)

struct RWMH <: AbstractMCMC.AbstractSampler
    σ::Float64
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::RWMH,
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
    sampler::RWMH,
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

    log_acceptance_ratio = min(0, logp_proposal - getlogp(state))

    if log(rand(rng)) < log_acceptance_ratio
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, getlogp(state))
    end
end

struct PriorMH <: AbstractMCMC.AbstractSampler
    prior_dist::Distribution
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::PriorMH,
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
    sampler::PriorMH,
    state::MHState,
    args...;
    kwargs...,
)
    params = getparams(state)
    proposal_dist = sampler.prior_dist
    proposal = rand(rng, proposal_dist)
    logp_proposal = only(
        LogDensityProblems.logdensity(logdensity_model.logdensity, proposal)
    )

    log_acceptance_ratio = min(
        0,
        logp_proposal - getlogp(state) + logpdf(proposal_dist, params) -
        logpdf(proposal_dist, proposal),
    )

    if log(rand(rng)) < log_acceptance_ratio
        return MHTransition(proposal), MHState(proposal, logp_proposal)
    else
        return MHTransition(params), MHState(params, getlogp(state))
    end
end

## tests

# for RWMH
# sample from Normal(10, 1)
struct NormalLogDensity end
LogDensityProblems.logdensity(l::NormalLogDensity, x) = logpdf(Normal(10, 1), only(x))
LogDensityProblems.dimension(l::NormalLogDensity) = 1
function LogDensityProblems.capabilities(::NormalLogDensity)
    return LogDensityProblems.LogDensityOrder{1}()
end

# for PriorMH
# sample from Categorical([0.2, 0.5, 0.3])
struct CategoricalLogDensity end
function LogDensityProblems.logdensity(l::CategoricalLogDensity, x)
    return logpdf(Categorical([0.2, 0.6, 0.2]), only(x))
end
LogDensityProblems.dimension(l::CategoricalLogDensity) = 1
function LogDensityProblems.capabilities(::CategoricalLogDensity)
    return LogDensityProblems.LogDensityOrder{0}()
end

## 

using StatsPlots

samples = AbstractMCMC.sample(
    Random.default_rng(), NormalLogDensity(), RWMH(1), 100000; initial_params=[0.0]
)
_samples = map(t -> only(t.params), samples)

histogram(_samples; normalize=:pdf, label="Samples", title="RWMH Sampling of Normal(10, 1)")
plot!(Normal(10, 1); linewidth=2, label="Ground Truth")

samples = AbstractMCMC.sample(
    Random.default_rng(),
    CategoricalLogDensity(),
    PriorMH(product_distribution([Categorical([0.3, 0.3, 0.4])])),
    100000;
    initial_params=[1],
)
_samples = map(t -> only(t.params), samples)

histogram(
    _samples;
    normalize=:probability,
    label="Samples",
    title="MH From Prior Sampling of Categorical([0.3, 0.3, 0.4])",
)
plot!(Categorical([0.2, 0.6, 0.2]); linewidth=2, label="Ground Truth")
