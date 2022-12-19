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
end
