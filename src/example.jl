using LogDensityProblems, Distributions, LinearAlgebra, Random
using OrderedCollections
## Define a simple GMM problem

struct GMM{Tdata}
    data::NamedTuple
end

struct ConditionedGMM{conditioned_vars}
    data::NamedTuple
    conditioned_values::NamedTuple{conditioned_vars}
end

function log_joint(;μ, w, z, x)
    # μ is mean of each component
    # w is weights of each component
    # z is assignment of each data point
    # x is data

    K = 2
    D = 2
    N = size(x, 1)
    logp = .0
    
    μ_prior = MvNormal(zeros(K), I)
    logp += sum(logpdf(μ_prior, μ))

    w_prior = Dirichlet(K, 1.0)
    logp += logpdf(w_prior, w)

    z_prior = Categorical(w)
    logp += sum([logpdf(z_prior, z[i]) for i in 1:N])

    for i in 1:N
        logp += logpdf(MvNormal(fill(μ[z[i]], D), I), x[i, :])
    end

    return logp
end

function condition(gmm::GMM, conditioned_values::NamedTuple)
    return ConditionedGMM(gmm.data, conditioned_values)
end

function logdensity(gmm::ConditionedGMM{conditioned_vars}, params) where {conditioned_vars}
    if conditioned_vars == (:μ, :w)
        return log_joint(;μ=gmm.conditioned_values.μ, w=gmm.conditioned_values.w, z=params.z, x=gmm.data)
    elseif conditioned_vars == (:z,)
        return log_joint(;μ=params.μ, w=params.w, z=gmm.conditioned_values.z, x=gmm.data)
    else
        throw(ArgumentError("condition group not supported"))
    end
end

function LogDensityProblems.logdensity(gmm::ConditionedGMM{conditioned_vars}, params_vec::AbstractVector) where {conditioned_vars}
    if conditioned_vars == (:μ, :w)
        params = (; z= params_vec)
    elseif conditioned_vars == (:z,)
        params = (; μ= params_vec[1:2], w= params_vec[3:4])
    else
        throw(ArgumentError("condition group not supported"))
    end

    return logdensity(gmm, params)
end

function LogDensityProblems.dimension(gmm::ConditionedGMM{conditioned_vars}) where {conditioned_vars}
    if conditioned_vars == (:μ, :w)
        return size(gmm.data.x, 1)
    elseif conditioned_vars == (:z,)
        return size(gmm.data.x, 1)
    else
        throw(ArgumentError("condition group not supported"))
    end
end

struct Gibbs <: AbstractMCMC.AbstractSampler 
    sampler_map::OrderedDict
end

# ! initialize the params here
struct GibbsState
    "contains all the values of the model parameters"
    values::NamedTuple
    states::OrderedDict
end

struct GibbsTransition
    values::NamedTuple
end

function AbstractMCMC.step(
    rng::AbstractRNG, model, sampler::Gibbs, args...; initial_params::NamedTuple, kwargs...
)
    states = OrderedDict()
    for group in collect(keys(sampler.sampler_map))
        sampler = sampler.sampler_map[group]
        cond_val = NamedTuple{group}([initial_params[g] for g in group]...)
        trans, state = AbstractMCMC.step(rng, condition(model, cond_val), sampler, args...; kwargs...)
        states[group] = state
    end
    return GibbsTransition(initial_params), GibbsState(initial_params, states)
end

# questions is: when do we assume the logp from last iteration is not reliable anymore

function AbstractMCMC.step(
    rng::AbstractRNG, model, sampler::Gibbs, state::GibbsState, args...; kwargs...
)
    for group in collect(keys(sampler.sampler_map))
        sampler = sampler.sampler_map[group]
        state = state.states[group]
        trans, state = AbstractMCMC.step(rng, condition(model, state.values[group]), sampler, state, args...; kwargs...)
        # TODO: what values to condition on here? stored where?
        state.states[group] = state
    end
    return 
end

# importance sampling
struct ImportanceSampling <: AbstractMCMC.AbstractSampler
    "number of samples"
    n::Int
    proposal
end

struct ImportanceSamplingState
    
end

struct ImportanceSamplingTransition
    values
end

# initial step
function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::ImportanceSampling, args...; kwargs...
)

end

function IS_step(rng::AbstractRNG, logdensity, sampler::ImportanceSampling, state::ImportanceSamplingState, args...; kwargs...)
    proposals = rand(rng, sampler.proposal, sampler.n)
    weights = logdensity.(proposals) .- log.(logpdf.(sampler.proposal, proposals))
    sample = rand(rng, Categorical(softmax(weights)))
    return ImportanceSamplingTransition(proposals[sample]), ImportanceSamplingState()
end


function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::ImportanceSampling, state::ImportanceSamplingState, args...; kwargs...
)
    return 
end

struct RWMH <: AbstractMCMC.AbstractSampler
    proposal
end

function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::RWMH, args...; kwargs...
)
    proposal = rand(rng, sampler.proposal)

    acceptance_probability = min(1, exp(logdensity(proposal) - logdensity(args[1])))
    if rand(rng) < acceptance_probability
        return proposal, nothing
    else
        return args[1], nothing
    end
end

function AbstractMCMC.step(
    rng::AbstractRNG, logdensity, sampler::RWMH, state::RWMHState, args...; kwargs...
)
    return 
end
