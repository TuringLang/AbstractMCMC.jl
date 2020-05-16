module AbstractMCMC

import BangBang
import ConsoleProgressMonitor
import LoggingExtras
import ProgressLogging
import StatsBase
import TerminalLoggers

import Distributed
import Logging
import Random

# Reexport sample
using StatsBase: sample
export sample

# Parallel sampling types
export MCMCThreads, MCMCDistributed

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
    AbstractMCMCParallel

An `AbstractMCMCParallel` algorithm represents a specific algorithm for sampling MCMC chains
in parallel.
"""
abstract type AbstractMCMCParallel end

"""
    MCMCThreads

The `MCMCThreads` algorithm allows to sample MCMC chains in parallel using multiple
threads.
"""
struct MCMCThreads <: AbstractMCMCParallel end

"""
    MCMCDistributed

The `MCMCDistributed` algorithm allows to sample MCMC chains in parallel using multiple
processes.
"""
struct MCMCDistributed <: AbstractMCMCParallel end

include("logging.jl")
include("interface.jl")
include("sample.jl")
include("stepper.jl")

end # module AbstractMCMC
