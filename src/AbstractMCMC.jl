module AbstractMCMC

import BangBang
import ConsoleProgressMonitor
import LoggingExtras
import ProgressLogging
import StatsBase
import TerminalLoggers
import Transducers

import Distributed
import Logging
import Random

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
    updatestate!!(state, transition_prev[, state_prev])

Return new instance of `state` using information from `transition_prev` and, optionally, `state_prev`.

Defaults to `setvalues!!(state, values(transition_prev))`.
"""
updatestate!!(state, transition_prev, state_prev) = updatestate!!(state, transition_prev)
updatestate!!(state, transition) = setvalues!!(state, Base.values(transition))

"""
    setvalues!!(state, values)

Update the values of the `state` with `values` and return it.

If `state` can be updated in-place, it is expected that this function returns `state` with updated
values. Otherwise a new `state` object with the new `values` is returned.
"""
function setvalues!! end

@doc """
    values(transition)

Return values in `transition`.
"""
Base.values


include("samplingstats.jl")
include("logging.jl")
include("interface.jl")
include("sample.jl")
include("stepper.jl")
include("transducer.jl")
include("deprecations.jl")

end # module AbstractMCMC
