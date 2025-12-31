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

# Support for extensions on Julia < 1.9
@static if !isdefined(Base, :get_extension)
    using Requires
end

# Reexport sample
using StatsBase: sample
export sample

# Parallel sampling types
export MCMCThreads, MCMCDistributed, MCMCSerial

# Callbacks
export MultiCallback, NameFilter

# Statistics Wrappers
export Skip, Thin, WindowStat

# TensorBoardCallback wrapper
export TensorBoardCallback

"""
    TensorBoardCallback(directory::String; kwargs...)
    TensorBoardCallback(logger; kwargs...)

Wrapper function to create a `TensorBoardCallback`. 
Requires `TensorBoardLogger` and `OnlineStats` to be loaded.

Returns an object that acts as a callback for `AbstractMCMC.step`.

# Keyword arguments
- `stats = nothing`: `OnlineStat` or Dict for custom statistics.
- `num_bins::Int = 100`: Number of bins for histograms.
- `filter = nothing`: Custom filter function `(varname, value) -> Bool`.
- `include = nothing`: Only log these parameter names.
- `exclude = nothing`: Don't log these parameter names.
- `include_extras::Bool = true`: Include extra statistics (log density, acceptance rate, etc.).
- `extras_filter = nothing`: Custom filter function for extras.
- `extras_include = nothing`: Only log these extra statistics.
- `extras_exclude = nothing`: Don't log these extra statistics.
- `include_hyperparams::Bool = false`: Include hyperparameters (logged once at start).
- `hyperparams_filter = nothing`: Custom filter function for hyperparameters.
- `hyperparams_include = nothing`: Only log these hyperparameters.
- `hyperparams_exclude = nothing`: Don't log these hyperparameters.
- `param_prefix::String = ""`: Prefix for logged parameter values.
- `extras_prefix::String = "extras/"`: Prefix for logged extra values.
"""
function TensorBoardCallback(args...; kwargs...)
    ext = if isdefined(Base, :get_extension)
        Base.get_extension(@__MODULE__, :AbstractMCMCTensorBoardLoggerExt)
    else
        if isdefined(@__MODULE__, :AbstractMCMCTensorBoardLoggerExt)
            AbstractMCMCTensorBoardLoggerExt
        else
            nothing
        end
    end

    if ext === nothing
        msg =
            "TensorBoardCallback requires TensorBoardLogger and OnlineStats to be loaded.\n" *
            "Please run `using TensorBoardLogger, OnlineStats`."
        error(msg)
    end

    return ext.TensorBoardCallback(args...; kwargs...)
end

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

# Backward compatibility for extensions on Julia < 1.9
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require TensorBoardLogger = "899adc3e-224a-11e9-021f-63837185c80f" begin
            @require OnlineStats = "a15396b6-48d5-5d58-9928-6d29437db91e" begin
                include("../ext/AbstractMCMCTensorBoardLoggerExt.jl")
            end
        end
    end
end

end # module AbstractMCMC
