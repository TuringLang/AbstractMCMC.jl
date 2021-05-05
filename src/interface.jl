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

"""
    bundle_samples(samples, model, sampler, state, chain_type[; kwargs...])

Bundle all `samples` that were sampled from the `model` with the given `sampler` in a chain.

The final `state` of the `sampler` can be included in the chain. The type of the chain can
be specified with the `chain_type` argument.

By default, this method returns `samples`.
"""
function bundle_samples(
    samples,
    ::AbstractModel,
    ::AbstractSampler,
    ::Any,
    ::Type;
    kwargs...
)
    return samples
end

function bundle_samples(
    samples::Vector,
    ::AbstractModel,
    ::AbstractSampler,
    ::Any,
    ::Type{Vector{T}};
    kwargs...
) where T
    return map(samples) do sample
        convert(T, sample)
    end
end

"""
    step(rng, model, sampler[, state; kwargs...])

Return a 2-tuple of the next sample and the next state of the MCMC `sampler` for `model`.

Samples describe the results of a single step of the `sampler`. As an example, a sample
might include a vector of parameters sampled from a prior distribution.

When sampling using [`sample`](@ref), every `step` call after the first has access to the
current `state` of the sampler.
"""
function step end

"""
    samples(sample, model, sampler[, N; kwargs...])

Generate a container for the samples of the MCMC `sampler` for the `model`, whose first
sample is `sample`.

The method can be called with and without a predefined number `N` of samples.
"""
function samples(
    sample,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...
)
    ts = Vector{typeof(sample)}(undef, 0)
    sizehint!(ts, N)
    return ts
end

function samples(
    sample,
    ::AbstractModel,
    ::AbstractSampler;
    kwargs...
)
    return Vector{typeof(sample)}(undef, 0)
end

"""
    save!!(samples, sample, iteration, model, sampler[, N; kwargs...])

Save the `sample` of the MCMC `sampler` at the current `iteration` in the container of
`samples`.

The function can be called with and without a predefined number `N` of samples. By default,
AbstractMCMC uses `push!!` from the Julia package
[BangBang](https://github.com/tkf/BangBang.jl) to append to the container, and widen its
type if needed.
"""
function save!!(
    samples::Vector,
    sample,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...
)
    s = BangBang.push!!(samples, sample)
    s !== samples && sizehint!(s, N)
    return s
end

function save!!(
    samples,
    sample,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler;
    kwargs...
)
    return BangBang.push!!(samples, sample)
end

# Deprecations
Base.@deprecate transitions(
    transition,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    kwargs...
) samples(transition, model, sampler, N; kwargs...) false
Base.@deprecate transitions(
    transition,
    model::AbstractModel,
    sampler::AbstractSampler;
    kwargs...
) samples(transition, model, sampler; kwargs...) false
