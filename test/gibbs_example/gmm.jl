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

function AbstractMCMC.condition(gmm::GMM, conditioned_values::NamedTuple)
    return ConditionedGMM(gmm.data, conditioned_values)
end

function LogDensityProblems.logdensity(
    gmm::ConditionedGMM{names}, params::AbstractVector
) where {names}
    if Set(names) == Set([:μ, :w]) # conditioned on μ, w, so params are z
        return log_joint(;
            μ=gmm.conditioned_values.μ, w=gmm.conditioned_values.w, z=params, x=gmm.data.x
        )
    elseif Set(names) == Set([:z, :w]) # conditioned on z, w, so params are μ
        return log_joint(;
            μ=params, w=gmm.conditioned_values.w, z=gmm.conditioned_values.z, x=gmm.data.x
        )
    elseif Set(names) == Set([:z, :μ]) # conditioned on z, μ, so params are w
        return log_joint(;
            μ=gmm.conditioned_values.μ, w=params, z=gmm.conditioned_values.z, x=gmm.data.x
        )
    else
        error("Unsupported conditioning configuration.")
    end
end

function LogDensityProblems.capabilities(::GMM)
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.capabilities(::ConditionedGMM)
    return LogDensityProblems.LogDensityOrder{0}()
end
