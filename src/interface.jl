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
    samples, model::AbstractModel, sampler::AbstractSampler, state, ::Type{T}; kwargs...
) where {T}
    # dispatch to internal method for default implementations to fix
    # method ambiguity issues (see #120)
    return _bundle_samples(samples, model, sampler, state, T; kwargs...)
end

function _bundle_samples(
    samples,
    @nospecialize(::AbstractModel),
    @nospecialize(::AbstractSampler),
    @nospecialize(::Any),
    ::Type;
    kwargs...,
)
    return samples
end
function _bundle_samples(
    samples::Vector,
    @nospecialize(::AbstractModel),
    @nospecialize(::AbstractSampler),
    @nospecialize(::Any),
    ::Type{Vector{T}};
    kwargs...,
) where {T}
    return map(samples) do sample
        convert(T, sample)
    end
end

"""
    step(rng, model, sampler[, state]; kwargs...)

Return a 2-tuple of the next sample and the next state of the MCMC `sampler` for `model`.

Samples describe the results of a single step of the `sampler`. As an example, a sample
might include a vector of parameters sampled from a prior distribution.

When sampling using [`sample`](@ref), every `step` call after the first has access to the
current `state` of the sampler.

## Keyword arguments

If the step being taken is going to be discarded (e.g. during burn-in, or if thinning is
performed), this method will be called with a `discard_sample=true` keyword argument.
Conversely, if the step being taken is to be retained, this method will be called with
`discard_sample=false`. This allows implementations of `step` to customize their behavior
based on whether or not the sample will be kept.

Other keyword arguments are passed through from the call to [`sample`](@ref). Because there
is no way of knowing in advance which keyword arguments will be passed, implementations of
`step` should include a `kwargs...` argument to capture any additional keyword arguments.
"""
function step end

"""
    step_warmup(rng, model, sampler[, state]; kwargs...)

Return a 2-tuple of the next sample and the next state of the MCMC `sampler` for `model`.

When sampling using [`sample`](@ref), this takes the place of [`AbstractMCMC.step`](@ref) in the first
`num_warmup` number of iterations, as specified by the `num_warmup` keyword to [`sample`](@ref).
This is useful if the sampler has an initial "warmup"-stage that is different from the
standard iteration.

By default, this defers to [`AbstractMCMC.step`](@ref), meaning that if a sampler does not
have special warmup behaviour, it only needs to implement `step`.

## Keyword arguments

The total number of warmup steps requested in sampling will be passed to the `step_warmup`
function as the `num_warmup` keyword argument. This allows implementations of `step_warmup`
to customise their behavior based on this information.

If the step being taken is going to be discarded (e.g. during burn-in, or if thinning is
performed), this method will be called with a `discard_sample=true` keyword argument.
Conversely, if the step being taken is to be retained, this method will be called with
`discard_sample=false`. This allows implementations of `step_warmup` to customize their
behavior based on whether or not the sample will be kept.

Other keyword arguments are passed through from the call to [`sample`](@ref). Because there
is no way of knowing in advance which keyword arguments will be passed, implementations of
`step_warmup` should include a `kwargs...` argument to capture any additional keyword
arguments.
"""
step_warmup(rng, model, sampler; kwargs...) = step(rng, model, sampler; kwargs...)
function step_warmup(rng, model, sampler, state; kwargs...)
    return step(rng, model, sampler, state; kwargs...)
end

"""
    samples(sample, model, sampler[, N; kwargs...])

Generate a container for the samples of the MCMC `sampler` for the `model`, whose first
sample is `sample`.

The method can be called with and without a predefined number `N` of samples.
"""
function samples(sample, ::AbstractModel, ::AbstractSampler, N::Integer; kwargs...)
    ts = Vector{typeof(sample)}(undef, 0)
    sizehint!(ts, N)
    return ts
end

function samples(sample, ::AbstractModel, ::AbstractSampler; kwargs...)
    return Vector{typeof(sample)}(undef, 0)
end

"""
    save!!(samples, sample, iteration, model, sampler[, N; kwargs...])

Save the `sample` of the MCMC `sampler` at the current `iteration` in the container of
`samples`.

The function can be called with and without a predefined number `N` of samples. By default,
AbstractMCMC uses `push!!` from the Julia package
[BangBang](https://github.com/JuliaFolds/BangBang.jl) to append to the container, and widen its
type if needed.
"""
function save!!(
    samples::Vector,
    sample,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...,
)
    s = BangBang.push!!(samples, sample)
    s !== samples && sizehint!(s, N)
    return s
end

function save!!(
    samples, sample, iteration::Integer, ::AbstractModel, ::AbstractSampler; kwargs...
)
    return BangBang.push!!(samples, sample)
end

# Deprecations
Base.@deprecate transitions(
    transition, model::AbstractModel, sampler::AbstractSampler, N::Integer; kwargs...
) samples(transition, model, sampler, N; kwargs...) false
Base.@deprecate transitions(
    transition, model::AbstractModel, sampler::AbstractSampler; kwargs...
) samples(transition, model, sampler; kwargs...) false
