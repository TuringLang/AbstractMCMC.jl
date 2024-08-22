using LogDensityProblems

abstract type AbstractHierNormal end

struct HierNormal <: AbstractHierNormal
    data::NamedTuple
end

struct ConditionedHierNormal{conditioned_vars} <: AbstractHierNormal
    data::NamedTuple
    conditioned_values::NamedTuple{conditioned_vars}
end

function log_joint(; mu, tau2, x)
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

function condition(hn::HierNormal, conditioned_values::NamedTuple)
    return ConditionedHierNormal(hn.data, conditioned_values)
end

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

function flatten(nt::NamedTuple)
    return only(values(nt))
end

function unflatten(vec::AbstractVector, group::Tuple)
    return NamedTuple((only(group) => vec,))
end

function recompute_logprob!!(hn::ConditionedHierNormal, vals, state)
    return setlogp!!(state, LogDensityProblems.logdensity(hn, vals))
end
