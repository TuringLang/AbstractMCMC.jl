module AbstractMCMC

import ConsoleProgressMonitor
import LoggingExtras
import ProgressLogging
import StatsBase
using StatsBase: sample
import TerminalLoggers

import Distributed
import Logging
import Random

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
    AbstractParallelAlgorithm

An `AbstractParallelAlgorithm` represents a specific algorithm for sampling MCMC chains in
parallel.
"""
abstract type AbstractParallelAlgorithm end

"""
    ParallelThreads

The `ParallelThreads` algorithm allows to sample MCMC chains in parallel using multiple
threads.
"""
struct ParallelThreads <: AbstractParallelAlgorithm end

"""
    ParallelDistributed

The `ParallelDistributed` algorithm allows to sample MCMC chains in parallel using multiple
cores.
"""
struct ParallelDistributed <: AbstractParallelAlgorithm end

include("logging.jl")
include("interface.jl")
include("sample.jl")
include("stepper.jl")

end # module AbstractMCMC
