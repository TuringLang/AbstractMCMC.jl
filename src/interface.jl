"""
    chainscat(c::AbstractChains)

Concatenate multiple chains.
"""
chainscat(c::AbstractChains...) = cat(c...; dims=3)

function bundle_samples(
    transitions,
    ::AbstractModel,
    ::AbstractSampler,
    ::Type;
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
    ts = Vector{typeof(transition)}(undef, 0)
    sizehint!(ts, N)
    return ts
end

function transitions(
    transition,
    ::AbstractModel,
    ::AbstractSampler;
    kwargs...
)
    return Vector{typeof(transition)}(undef, 0)
end

"""
    save!!(transitions, transition, iteration, model, sampler, N[; kwargs...])
    save!!(transitions, transition, iteration, model, sampler[; kwargs...])

Save the `transition` of the MCMC `sampler` at the current `iteration` in the container of
`transitions`.

The function can be called with and without a predefined size `N`. By default, AbstractMCMC
uses ``push!!`` from the Julia package [BangBang](https://github.com/tkf/BangBang.jl) to
append to the container, and widen its type if needed.
"""
function save!!(
    transitions::Vector,
    transition,
    iteration::Integer,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...
)
    new_ts = BangBang.push!!(transitions, transition)
    new_ts !== transitions && sizehint!(new_ts, N)
    return new_ts
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
