module AbstractMCMC

using BangBang: BangBang
using ConsoleProgressMonitor: ConsoleProgressMonitor
using LogDensityProblems: LogDensityProblems
using LoggingExtras: LoggingExtras
using ProgressLogging: ProgressLogging
using StatsBase: StatsBase
using TerminalLoggers: TerminalLoggers
using Transducers: Transducers
using FillArrays: FillArrays

using Distributed: Distributed
using Logging: Logging
using Random: Random
using UUIDs: UUIDs

# Reexport sample
using StatsBase: sample
export sample

# Parallel sampling types
export MCMCThreads, MCMCDistributed, MCMCSerial

# Callbacks
export MultiCallback, NameFilter

# TensorBoard integration - returns TensorBoardCallback type when extension is loaded
function TensorBoardCallback(args...; kwargs...)
    ext = Base.get_extension(@__MODULE__, :AbstractMCMCTensorBoardLoggerExt)
    if ext === nothing
        error(
            "TensorBoardCallback requires TensorBoardLogger and OnlineStats to be loaded. " *
            "Add `using TensorBoardLogger, OnlineStats` before using TensorBoardCallback.",
        )
    end
    if !isdefined(@__MODULE__, :Skip)
        eval(:(const Skip = $(ext.Skip)))
        eval(:(const Thin = $(ext.Thin)))
        eval(:(const WindowStat = $(ext.WindowStat)))
        eval(:(export Skip, Thin, WindowStat))
    end
    return ext.TensorBoardCallback(args...; kwargs...)
end
export TensorBoardCallback

"""
    AbstractChains

`AbstractChains` is an abstract type for an object that stores
parameter samples generated through a MCMC process.
"""
abstract type AbstractChains end

"""
    AbstractMCMC.from_samples(::Type{T}, samples::Matrix) where {T<:AbstractChains}

Convert a Matrix of parameter samples to an `AbstractChains` object.

Methods of this function should be implemented with the signature listed above, and should
return a chain of type `T`.

This function is the inverse of [`to_samples`](@ref).

In general, it is expected that for `chains::Tchn`, `from_samples(Tchn, to_samples(Tsample,
chains))` contains data that are equivalent to that in `chains` (although differences in
e.g. metadata or ordering are permissible).

Furthermore, the same should hold true for `to_samples(Tsample, from_samples(Tchn,
samples))` where `samples::Matrix{Tsample}`.
"""
function from_samples end

"""
    AbstractMCMC.to_samples(::Type{T}, chains::AbstractChains) where {T}

Convert an `AbstractChains` object to an `Matrix` of parameter samples.

Methods of this function should be implemented with the signature listed above, and should
return a `Matrix{T}`.

See also: [`from_samples`](@ref).
"""
function to_samples end

"""
    AbstractSampler

The `AbstractSampler` type is intended to be inherited from when
implementing a custom sampler. Any persistent state information should be
saved in a subtype of `AbstractSampler`.

When defining a new sampler, you should also overload the function
`transition_type`, which tells the `sample` function what type of parameter
it should expect to receive.
"""
abstract type AbstractSampler end

"""
    AbstractModel

An `AbstractModel` represents a generic model type that can be used to perform inference.
"""
abstract type AbstractModel end

"""
    AbstractMCMCEnsemble

An `AbstractMCMCEnsemble` algorithm represents a specific algorithm for sampling MCMC chains
in parallel.
"""
abstract type AbstractMCMCEnsemble end

"""
    MCMCThreads

The `MCMCThreads` algorithm allows users to sample MCMC chains in parallel using multiple
threads.

# Usage

```julia
sample(model, sampler, MCMCThreads(), N, nchains)
```

See also [`sample`](@ref).
"""
struct MCMCThreads <: AbstractMCMCEnsemble end

"""
    MCMCDistributed

The `MCMCDistributed` algorithm allows users to sample MCMC chains in parallel using multiple
processes.

# Usage

```julia
sample(model, sampler, MCMCDistributed(), N, nchains)
```

See also [`sample`](@ref).
"""
struct MCMCDistributed <: AbstractMCMCEnsemble end

"""
    MCMCSerial

The `MCMCSerial` algorithm allows users to sample serially, with no thread or process parallelism.

# Usage

```julia
sample(model, sampler, MCMCSerial(), N, nchains)
```

See also [`sample`](@ref).
"""
struct MCMCSerial <: AbstractMCMCEnsemble end

"""
    requires_unconstrained_space(sampler::AbstractSampler)::Bool

Return `true` if the given sampler must run in unconstrained space. Defaults to true.
"""
requires_unconstrained_space(::AbstractSampler) = true

"""
    getparams([model::AbstractModel, ]state)::Vector{<:Real}

Retrieve the values of parameters from the sampler's `state` as a `Vector{<:Real}`.
"""
function getparams end

function getparams(model::AbstractModel, state)
    return getparams(state)
end

"""
    getstats(state)::NamedTuple

Retrieve sampler statistics from the sampler's `state` as a `NamedTuple`.
"""
function getstats end

"""
    setparams!!([model::AbstractModel, ]state, params)

Set the values of parameters in the sampler's `state` from a `Vector{<:Real}`. 

This function should follow the `BangBang` interface: mutate `state` in-place if possible and 
return the mutated `state`. Otherwise, it should return a new `state` containing the updated parameters.

Although not enforced, it should hold that `setparams!!(state, getparams(state)) == state`. In other
words, the sampler should implement a consistent transformation between its internal representation
and the vector representation of the parameter values.

Sometimes, to maintain the consistency of the log density and parameter values, a `model`
should be provided. This is useful for samplers that need to evaluate the log density at the new parameter values.
"""
function setparams!! end

function setparams!!(model::AbstractModel, state, params)
    return setparams!!(state, params)
end

include("samplingstats.jl")
include("logging.jl")
include("interface.jl")
include("sample.jl")
include("stepper.jl")
include("transducer.jl")
include("logdensityproblems.jl")
include("callbacks.jl")

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, _
            if (
                any(a -> a <: LogDensityModel, argtypes) &&
                exc.f isa Function &&
                Base.parentmodule(exc.f) == LogDensityProblems
            )
                print(
                    io,
                    "\n`AbstractMCMC.LogDensityModel` is a wrapper and does not itself implement the LogDensityProblems.jl interface. To use LogDensityProblems.jl methods, access the inner type with (e.g.) `logdensity(model.logdensity, params)` instead of `logdensity(model, params)`.",
                )
            end
        end
    end
end

end # module AbstractMCMC
