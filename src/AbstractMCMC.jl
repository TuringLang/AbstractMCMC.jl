module AbstractMCMC

import ProgressLogging
import StatsBase
using StatsBase: sample

import Distributed
import Logging
using Random: GLOBAL_RNG, AbstractRNG, seed!
import UUIDs

"""
    AbstractChains

`AbstractChains` is an abstract type for an object that stores
parameter samples generated through a MCMC process.
"""
abstract type AbstractChains end

chainscat(c::AbstractChains...) = cat(c...; dims=3)

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
    sample([rng, ]model, sampler, N; kwargs...)

Return `N` samples from the MCMC `sampler` for the provided `model`.

If a callback function `f` with type signature 
```julia
f(rng::AbstractRNG, model::AbstractModel, sampler::AbstractSampler, N::Integer,
  iteration::Integer, transition; kwargs...)
```
may be provided as keyword argument `callback`. It is called after every sampling step.
"""
function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    kwargs...
)
    return sample(GLOBAL_RNG, model, sampler, N; kwargs...)
end

function StatsBase.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    progress = true,
    progressname = "Sampling",
    callback = (args...; kwargs...) -> nothing,
    chain_type::Type=Any,
    kwargs...
)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")

    # Perform any necessary setup.
    sample_init!(rng, model, sampler, N; kwargs...)

    # Create a progress bar.
    if progress
        progressid = UUIDs.uuid4()
        Logging.@logmsg(ProgressLogging.ProgressLevel, progressname, progress=NaN,
                        _id=progressid)
    end

    local transitions
    try
        # Obtain the initial transition.
        transition = step!(rng, model, sampler, N; iteration=1, kwargs...)

        # Run callback.
        callback(rng, model, sampler, N, 1, transition; kwargs...)

        # Save the transition.
        transitions = transitions_init(transition, model, sampler, N; kwargs...)
        transitions_save!(transitions, 1, transition, model, sampler, N; kwargs...)

        # Update the progress bar.
        if progress
            Logging.@logmsg(ProgressLogging.ProgressLevel, progressname, progress=1/N,
                            _id=progressid)
        end

        # Step through the sampler.
        for i in 2:N
            # Obtain the next transition.
            transition = step!(rng, model, sampler, N, transition; iteration=i, kwargs...)

            # Run callback.
            callback(rng, model, sampler, N, i, transition; kwargs...)

            # Save the transition.
            transitions_save!(transitions, i, transition, model, sampler, N; kwargs...)

            # Update the progress bar.
            if progress
                Logging.@logmsg(ProgressLogging.ProgressLevel, progressname, progress=i/N,
                                _id=progressid)
            end
        end
    finally
        # Close the progress bar.
        if progress
            Logging.@logmsg(ProgressLogging.ProgressLevel, progressname, progress="done",
                            _id=progressid)
        end
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, model, sampler, N, transitions; kwargs...)

    return bundle_samples(rng, model, sampler, N, transitions, chain_type; kwargs...)
end

"""
    sample_init!(rng, model, sampler, N[; kwargs...])

Perform the initial setup of the MCMC `sampler` for the provided `model`.

This function is not intended to return any value, any set up should mutate the `sampler`
or the `model` in-place. A common use for `sample_init!` might be to instantiate a particle
field for later use, or find an initial step size for a Hamiltonian sampler.
"""
function sample_init!(
    ::AbstractRNG,
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
    ::AbstractRNG,
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
    ::AbstractRNG, 
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
    ::AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::Integer = 1,
    transition = nothing;
    kwargs...
)
    error("function `step!` is not implemented for models of type $(typeof(model)), ",
        "samplers of type $(typeof(sampler)), and transitions of type $(typeof(transition))")
end

"""
    transitions_init(transition, model, sampler, N[; kwargs...])

Generate a container for the `N` transitions of the MCMC `sampler` for the provided
`model`, whose first transition is `transition`.
"""
function transitions_init(
    transition,
    ::AbstractModel,
    ::AbstractSampler,
    N::Integer;
    kwargs...
)
    return Vector{typeof(transition)}(undef, N)
end

"""
    transitions_save!(transitions, iteration, transition, model, sampler, N[; kwargs...])

Save the `transition` of the MCMC `sampler` at the current `iteration` in the container of
`transitions`.
"""
function transitions_save!(
    transitions::AbstractVector,
    iteration::Integer,
    transition,
    ::AbstractModel,
    ::AbstractSampler,
    ::Integer;
    kwargs...
)
    transitions[iteration] = transition
    return
end

"""
    psample([rng::AbstractRNG, ]model::AbstractModel, sampler::AbstractSampler, N::Integer,
            nchains::Integer; kwargs...)

Sample `nchains` chains using the available threads, and combine them into a single chain.

By default, the random number generator, the model and the samplers are deep copied for each
thread to prevent contamination between threads. 
"""
function psample(
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return psample(GLOBAL_RNG, model, sampler, N, nchains; kwargs...)
end

function psample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer,
    nchains::Integer;
    progress = true,
    progressname = "Parallel sampling",
    kwargs...
)
    # Copy the random number generator, model, and sample for each thread
    rngs = [deepcopy(rng) for _ in 1:Threads.nthreads()]
    models = [deepcopy(model) for _ in 1:Threads.nthreads()]
    samplers = [deepcopy(sampler) for _ in 1:Threads.nthreads()]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    # Create a progress bar and a channel for progress logging.
    if progress
        progressid = UUIDs.uuid4()
        Logging.@logmsg(ProgressLogging.ProgressLevel, progressname, progress=NaN,
                        _id=progressid)
        channel = Distributed.RemoteChannel(() -> Channel{Bool}(nchains), 1)
    end

    try
        Distributed.@sync begin
            if progress
                Distributed.@async begin
                    # Update the progress bar.
                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        Logging.@logmsg(ProgressLogging.ProgressLevel, progressname,
                                        progress=progresschains/nchains, _id=progressid)
                    end
                end
            end

            Distributed.@async begin
                Threads.@threads for i in 1:nchains
                    # Obtain the ID of the current thread.
                    id = Threads.threadid()

                    # Seed the thread-specific random number generator with the pre-made seed.
                    subrng = rngs[id]
                    seed!(subrng, seeds[i])
  
                    # Sample a chain and save it to the vector.
                    chains[i] = sample(subrng, models[id], samplers[id], N;
                                       progress = false, kwargs...)

                    # Update the progress bar.
                    progress && put!(channel, true)
                end

                # Stop updating the progress bar.
                progress && put!(channel, false)
            end
        end
    finally
        # Close the progress bar.
        if progress
            Logging.@logmsg(ProgressLogging.ProgressLevel, progressname,
                            progress="done", _id=progressid)
        end
    end

    # Concatenate the chains together.
    return reduce(chainscat, chains)
end


##################
# Iterator tools #
##################
struct Stepper{A<:AbstractRNG, ModelType<:AbstractModel, SamplerType<:AbstractSampler, K}
    rng::A
    model::ModelType
    s::SamplerType
    kwargs::K
end

function Base.iterate(stp::Stepper, state=nothing)
    t = step!(stp.rng, stp.model, stp.s, 1, state; stp.kwargs...)
    return t, t
end

Base.IteratorSize(::Type{<:Stepper}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:Stepper}) = Base.EltypeUnknown()

"""
    steps!([rng::AbstractRNG, ]model::AbstractModel, s::AbstractSampler, kwargs...)

`steps!` returns an iterator that returns samples continuously, after calling `sample_init!`.

Usage:

```julia
for transition in steps!(MyModel(), MySampler())
    println(transition)

    # Do other stuff with transition below.
end
```
"""
function steps!(
    model::AbstractModel,
    s::AbstractSampler,
    kwargs...
)
    return steps!(GLOBAL_RNG, model, s; kwargs...)
end

function steps!(
    rng::AbstractRNG,
    model::AbstractModel,
    s::AbstractSampler,
    kwargs...
)
    sample_init!(rng, model, s, 0)
    return Stepper(rng, model, s, kwargs)
end

##################################
# Sample-until-convergence tools #
##################################

struct StopException <: Exception end

"""
    sample([rng::AbstractRNG, ]model::AbstractModel, s::AbstractSampler, is_done::Function; kwargs...)

`sample` will continuously draw samples without defining a maximum number of samples until
a convergence criteria defined by a user-defined function `is_done` returns `true`.

`is_done` is a function `f` that returns a `Bool`, with the signature

```julia
f(rng::AbstractRNG, model::AbstractModel, s::AbstractSampler, transitions::Vector, iteration::Int; kwargs...)
```

`is_done` should return `true` when sampling should end, and `false` otherwise.
"""
function StatsBase.sample(
    model::AbstractModel,
    s::AbstractSampler;
    kwargs...
)
    return sample(GLOBAL_RNG, model, s, kwargs...)
end

function StatsBase.sample(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    is_done::Function;
    chain_type::Type=Any,
    callback = (args...; kwargs...) -> nothing,
    kwargs...
)
    # Obtain the initial transition.
    transition = step!(rng, model, sampler, 1; iteration=1, kwargs...)

    # Run callback.
    callback(rng, model, sampler, 1, 1, transition; kwargs...)

    # Save the transition.
    transitions = [transition]

    # Step through the sampler until stopping.
    i = 2
    done = false

    while !is_done(rng, model, sampler, transitions, i; kwargs...)
        # Obtain the next transition.
        transition = step!(rng, model, sampler, 1, transition; iteration=i, kwargs...)

        # Run callback.
        callback(rng, model, sampler, 1, i, transition; kwargs...)

        # Save the transition.
        push!(transitions, transition)

        # Check transition.
        done = is_done(rng, model, sampler, transitions, i; kwargs...)

        # Increment iteration counter.
        i += 1
    end

    # Wrap the samples up.
    return bundle_samples(rng, model, sampler, i, transitions, chain_type; kwargs...)
end

end # module AbstractMCMC
