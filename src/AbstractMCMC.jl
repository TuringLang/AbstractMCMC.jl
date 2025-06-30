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

"""
    AbstractChains

`AbstractChains` is an abstract type for an object that stores
parameter samples generated through a MCMC process.
"""
abstract type AbstractChains end

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
"""
struct MCMCThreads <: AbstractMCMCEnsemble end

"""
    MCMCDistributed

The `MCMCDistributed` algorithm allows users to sample MCMC chains in parallel using multiple
processes.
"""
struct MCMCDistributed <: AbstractMCMCEnsemble end

"""
    MCMCSerial

The `MCMCSerial` algorithm allows users to sample serially, with no thread or process parallelism.
"""
struct MCMCSerial <: AbstractMCMCEnsemble end

"""
    getparams([model::AbstractModel, ]state)

Retrieve the values of parameters from the sampler's `state` as a `Vector{<:Real}`.
"""
function getparams end

function getparams(model::AbstractModel, state)
    return getparams(state)
end

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
