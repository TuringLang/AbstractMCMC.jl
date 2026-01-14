"""
    LogDensityModel <: AbstractMCMC.AbstractModel

Wrapper around something that implements the LogDensityProblems.jl interface.

Note that this does _not_ implement the LogDensityProblems.jl interface itself,
but it is simply useful for indicating to the `sample` and other `AbstractMCMC` methods
that the wrapped object implements the LogDensityProblems.jl interface.

# Fields
- `logdensity`: The object that implements the LogDensityProblems.jl interface.
"""
struct LogDensityModel{L} <: AbstractModel
    logdensity::L
    function LogDensityModel{L}(logdensity::L) where {L}
        if LogDensityProblems.capabilities(logdensity) === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        return new{L}(logdensity)
    end
end

LogDensityModel(logdensity::L) where {L} = LogDensityModel{L}(logdensity)

# Fallbacks: Wrap log density function in a model
"""
    sample(
        rng::Random.AbstractRNG=Random.default_rng(),
        logdensity,
        sampler::AbstractSampler,
        N_or_isdone;
        kwargs...,
    )

Wrap the `logdensity` function in a [`LogDensityModel`](@ref), and call `sample` with the resulting model instead of `logdensity`.

The `logdensity` function has to support the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG, logdensity, sampler::AbstractSampler, N_or_isdone; kwargs...
)
    return StatsBase.sample(rng, _model(logdensity), sampler, N_or_isdone; kwargs...)
end

"""
    sample(
        rng::Random.AbstractRNG=Random.default_rng(),
        logdensity,
        sampler::AbstractSampler,
        parallel::AbstractMCMCEnsemble,
        N::Integer,
        nchains::Integer;
        kwargs...,
    )

Wrap the `logdensity` function in a [`LogDensityModel`](@ref), and call `sample` with the resulting model instead of `logdensity`.

The `logdensity` function has to support the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    logdensity,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...,
)
    return StatsBase.sample(
        rng, _model(logdensity), sampler, parallel, N, nchains; kwargs...
    )
end

"""
    steps(
        rng::Random.AbstractRNG=Random.default_rng(),
        logdensity,
        sampler::AbstractSampler;
        kwargs...,
    )

Wrap the `logdensity` function in a [`LogDensityModel`](@ref), and call `steps` with the resulting model instead of `logdensity`.

The `logdensity` function has to support the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
function steps(rng::Random.AbstractRNG, logdensity, sampler::AbstractSampler; kwargs...)
    return steps(rng, _model(logdensity), sampler; kwargs...)
end

"""
    Sample(
        rng::Random.AbstractRNG=Random.default_rng(),
        logdensity,
        sampler::AbstractSampler;
        kwargs...,
    )

Wrap the `logdensity` function in a [`LogDensityModel`](@ref), and call `Sample` with the resulting model instead of `logdensity`.

The `logdensity` function has to support the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
function Sample(rng::Random.AbstractRNG, logdensity, sampler::AbstractSampler; kwargs...)
    return Sample(rng, _model(logdensity), sampler; kwargs...)
end

function _model(logdensity)
    if LogDensityProblems.capabilities(logdensity) === nothing
        throw(
            ArgumentError(
                "the log density function does not support the LogDensityProblems.jl interface. Please implement the interface or provide a model of type `AbstractMCMC.AbstractModel`",
            ),
        )
    end
    return LogDensityModel(logdensity)
end
