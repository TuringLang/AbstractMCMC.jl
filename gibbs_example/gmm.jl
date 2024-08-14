using LogDensityProblems

abstract type AbstractGMM end

struct GMM <: AbstractGMM
    data::NamedTuple
end

struct ConditionedGMM{conditioned_vars} <: AbstractGMM
    data::NamedTuple
    conditioned_values::NamedTuple{conditioned_vars}
end

function log_joint(; μ, w, z, x)
    # μ is mean of each component
    # w is weights of each component
    # z is assignment of each data point
    # x is data

    K = 2 # assume we know the number of components
    D = 2 # dimension of each data point
    N = size(x, 2) # number of data points
    logp = 0.0

    μ_prior = MvNormal(zeros(K), I)
    logp += logpdf(μ_prior, μ)

    w_prior = Dirichlet(K, 1.0)
    logp += logpdf(w_prior, w)

    z_prior = Categorical(w)
    logp += sum([logpdf(z_prior, z[i]) for i in 1:N])

    obs_priors = [MvNormal(fill(μₖ, D), I) for μₖ in μ]
    for i in 1:N
        logp += logpdf(obs_priors[z[i]], x[:, i])
    end

    return logp
end

function condition(gmm::GMM, conditioned_values::NamedTuple)
    return ConditionedGMM(gmm.data, conditioned_values)
end

function _logdensity(gmm::ConditionedGMM{(:μ, :w)}, params)
    return log_joint(;
        μ=gmm.conditioned_values.μ, w=gmm.conditioned_values.w, z=params.z, x=gmm.data.x
    )
end
function _logdensity(gmm::ConditionedGMM{(:z,)}, params)
    return log_joint(; μ=params.μ, w=params.w, z=gmm.conditioned_values.z, x=gmm.data.x)
end

function LogDensityProblems.logdensity(
    gmm::ConditionedGMM{(:μ, :w)}, params_vec::AbstractVector
)
    return _logdensity(gmm, (; z=params_vec))
end
function LogDensityProblems.logdensity(
    gmm::ConditionedGMM{(:z,)}, params_vec::AbstractVector
)
    return _logdensity(gmm, (; μ=params_vec[1:2], w=params_vec[3:4]))
end

function LogDensityProblems.dimension(gmm::ConditionedGMM{(:μ, :w)})
    return size(gmm.data.x, 1)
end
function LogDensityProblems.dimension(gmm::ConditionedGMM{(:z,)})
    return size(gmm.data.x, 1)
end

## test using Turing

# data generation

using Distributions
using FillArrays
using LinearAlgebra
using Random

w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = Distributions.MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

N = 60
x = rand(mixturemodel, N);

# Turing model from https://turinglang.org/docs/tutorials/01-gaussian-mixture-model/
using Turing

@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return μ, w, k, __varinfo__
end

model = gaussian_mixture_model(x);

using Test
# full model
μ, w, k, vi = model()
@test log_joint(; μ=μ, w=w, z=k, x=x) ≈ DynamicPPL.getlogp(vi)

gmm = GMM((; x=x))

# cond model on μ, w
μ, w, k, vi = (DynamicPPL.condition(model, (μ=μ, w=w)))()
@test _logdensity(condition(gmm, (; μ=μ, w=w)), (; z=k)) ≈ DynamicPPL.getlogp(vi)

# cond model on z
μ, w, k, vi = (DynamicPPL.condition(model, (z = k)))()
@test _logdensity(condition(gmm, (; z=k)), (; μ=μ, w=w)) ≈ DynamicPPL.getlogp(vi)
