# AbstractChains interface
#
# NOTE: The entire interface is treated as experimental except for the AbstractChains type
# itself, along with `chainscat` and `chainsstack`. Thus, if you change any of those three,
# it is mandatory to release a breaking version. Other changes to the AbstractChains
# interface can be made in patch releases.

"""
    AbstractMCMC.AbstractChains

An abstract type for Markov chains, i.e., a data structure which stores samples
obtained from Markov chain Monte Carlo (MCMC) sampling.

!!! danger "Explicitly experimental"

    Although the abstract type `AbstractMCMC.AbstractChains` itself, along with the
    functions `chainscat` and `chainsstack`, are exported and public, please note that *all
    other parts of the interface remain experimental and subject to change*. In particular,
    breaking changes to the interface may be introduced in formally non-breaking releases.

Markov chains should generally have dictionary-like behaviour, where keys are mapped to
matrices of values.

## Interface

To implement a new subtype of `AbstractChains`, you need to define the following methods:

- `Base.size` should return a tuple of ints (the exact meaning is left to you)
- `Base.keys` should return a list of keys
- [`AbstractMCMC.get_data`](@ref)`(chn, key)`
- [`AbstractMCMC.iter_indices`](@ref)`(chn)`
- [`AbstractMCMC.chain_indices`](@ref)`(chn)`

You can optionally define the following methods for efficiency:

- [`AbstractChains.niters`](@ref)`(chn)`
- [`AbstractChains.nchains`](@ref)`(chn)`
"""
abstract type AbstractChains end

"""
    chainscat(c::AbstractChains...)

Concatenate multiple chains.

By default, the chains are concatenated along the third dimension by calling
`cat(c...; dims=3)`.
"""
chainscat(c::AbstractChains...) = cat(c...; dims=3)

"""
    chainsstack(c::AbstractVector)

Stack chains in `c`.

By default, the vector of chains is returned unmodified. If `eltype(c) <: AbstractChains`,
then `reduce(chainscat, c)` is called.
"""
chainsstack(c) = c
chainsstack(c::AbstractVector{<:AbstractChains}) = reduce(chainscat, c)
include("experimental/chains.jl")
