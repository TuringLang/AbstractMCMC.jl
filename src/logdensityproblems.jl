"""
    LogDensityModel <: AbstractMCMC.AbstractModel

Wrapper around something that implements the LogDensityProblem.jl interface.

Note that this does _not_ implement the LogDensityProblems.jl interface itself,
but it simply useful for indicating to the `sample` and other `AbstractMCMC` methods
that the wrapped object implements the LogDensityProblems.jl interface.

# Fields
- `logdensity`: The object that implements the LogDensityProblems.jl interface.
"""
struct LogDensityModel{L} <: AbstractModel
    logdensity::L
    function LogDensityModel{L}(logdensity::L) where {L}
        if LogDensityProblems.capabilities(logdensity) === nothing
            throw(ArgumentError("The log density function does not support the LogDensityProblems.jl interface"))
        end
        return new{L}(logdensity)
    end
end

LogDensityModel(logdensity::L) where {L} = LogDensityModel{L}(logdensity)
