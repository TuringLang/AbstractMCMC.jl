"""
    chainscat(c::AbstractChains)

Concatenate multiple chains.
"""
chainscat(c::AbstractChains...) = cat(c...; dims=3)

"""
    sample_init!(rng, model, sampler, N[; kwargs...])

Perform the initial setup of the MCMC `sampler` for the provided `model`.

This function is not intended to return any value, any set up should mutate the `sampler`
or the `model` in-place. A common use for `sample_init!` might be to instantiate a particle
field for later use, or find an initial step size for a Hamiltonian sampler.
"""
function sample_init!(
    ::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::Integer;
    kwargs...
)
    @debug "the default `sample_init!` function is used" typeof(model) typeof(sampler)
    return
end

"""
    sample_end!(rng, model, sampler, N, transitions[; kwargs...])

Perform final modifications after sampling from the MCMC `sampler` for the provided `model`,
resulting in the provided `transitions`.

This function is not intended to return any value, any set up should mutate the `sampler`
or the `model` in-place.

This function is useful in cases where you might want to transform the `transitions`,
save the `sampler` to disk, or perform any clean-up or finalization.
"""
function sample_end!(
    ::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::Integer,
    transitions;
    kwargs...
)
    @debug "the default `sample_end!` function is used" typeof(model) typeof(sampler) typeof(transitions)
    return
end

function bundle_samples(
    ::Random.AbstractRNG,
    ::AbstractModel,
    ::AbstractSampler,
    ::Integer,
    transitions,
    ::Type{Any};
    kwargs...
)
    return transitions
end

"""
    step!(rng, model, sampler[, N = 1, transition = nothing; kwargs...])

Return the transition for the next step of the MCMC `sampler` for the provided `model`,
using the provided random number generator `rng`.

Transitions describe the results of a single step of the `sampler`. As an example, a
transition might include a vector of parameters sampled from a prior distribution.

The `step!` function may modify the `model` or the `sampler` in-place. For example, the
`sampler` may have a state variable that contains a vector of particles or some other value
that does not need to be included in the returned transition.

When sampling from the `sampler` using [`sample`](@ref), every `step!` call after the first
has access to the previous `transition`. In the first call, `transition` is set to `nothing`.
"""
function step!(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer = 1;
    kwargs...
)
    return step!(rng, model, sampler, N, nothing; kwargs...)
end

"""
    transitions(transition, model, sampler, N[; kwargs...])
    transitions(transition, model, sampler[; kwargs...])

Generate a container for the `N` transitions of the MCMC `sampler` for the provided
`model`, whose first transition is `transition`.

The method can be called with and without a predefined size `N`.
"""
function transitions(
    transition,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...
)
    return Vector{typeof(transition)}(undef, N)
end

function transitions(
    transition,
    ::AbstractModel,
    ::AbstractSampler;
    kwargs...
)
    return Vector{typeof(transition)}(undef, 1)
end

"""
    save!!(transitions, transition, iteration, model, sampler, N[; kwargs...])
    save!!(transitions, transition, iteration, model, sampler[; kwargs...])

Save the `transition` of the MCMC `sampler` at the current `iteration` in the container of
`transitions`.

The function can be called with and without a predefined size `N`. By default, AbstractMCMC
uses ``setindex!`` and ``push!!`` from the Julia package
[BangBang](https://github.com/tkf/BangBang.jl) to write to and append to the container,
and widen the container type if needed.
"""
function save!!(
    transitions,
    transition,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler,
    ::Integer;
    kwargs...
)
    return BangBang.setindex!!(transitions, transition, iteration)
end

function save!!(
    transitions,
    transition,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler;
    kwargs...
)
    return BangBang.push!!(transitions, transition)
end

Base.@deprecate transitions_init(transition, model::AbstractModel, sampler::AbstractSampler, N::Integer; kwargs...) transitions(transition, model, sampler, N; kwargs...) false
Base.@deprecate transitions_init(transition, model::AbstractModel, sampler::AbstractSampler; kwargs...) transitions(transition, model, sampler; kwargs...) false
Base.@deprecate transitions_save!(transitions, iteration::Integer, transition, model::AbstractModel, sampler::AbstractSampler; kwargs...) save!!(transitions, transition, iteration, model, sampler; kwargs...) false
Base.@deprecate transitions_save!(transitions, iteration::Integer, transition, model::AbstractModel, sampler::AbstractSampler, N::Integer; kwargs...) save!!(transitions, transition, iteration, model, sampler, N; kwargs...) false

