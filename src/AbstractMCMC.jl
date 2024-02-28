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
    updatestate!!(model, state, transition_prev[, state_prev])

Return new instance of `state` using information from `model`, `transition_prev` and, optionally, `state_prev`.

Defaults to `realize!!(state, realize(transition_prev))`.
"""
function updatestate!!(model, state, transition_prev, state_prev)
    return updatestate!!(state, transition_prev)
end
updatestate!!(model, state, transition) = realize!!(state, realize(transition))

"""
    realize!!(state, realization)

Update the realization of the `state` with `realization` and return it.

If `state` can be updated in-place, it is expected that this function returns `state` with updated
realize. Otherwise a new `state` object with the new `realization` is returned.
"""
function realize!! end

"""
    realize(transition)

Return the realization of the random variables present in `transition`.
"""
function realize end


include("samplingstats.jl")
include("logging.jl")
include("interface.jl")
include("sample.jl")
include("stepper.jl")
include("transducer.jl")
include("logdensityproblems.jl")

end # module AbstractMCMC
