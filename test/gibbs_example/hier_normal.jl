using AbstractPPL: AbstractPPL

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

# `mu` and `tau2` are length-1 vectors to make 
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

function AbstractPPL.condition(hn::HierNormal, conditioned_values::NamedTuple)
    return ConditionedHierNormal(hn.data, conditioned_values)
end

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
