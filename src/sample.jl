# Default implementations of `sample`.

function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    arg;
    kwargs...
)
    return StatsBase.sample(Random.GLOBAL_RNG, model, sampler, arg; kwargs...)
end

function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    arg;
    kwargs...
)
    return mcmcsample(rng, model, sampler, arg; kwargs...)
end

function StatsBase.sample(
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCParallel,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return StatsBase.sample(Random.GLOBAL_RNG, model, sampler, parallel, N, nchains;
                            kwargs...)
end

function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCParallel,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    return mcmcsample(rng, model, sampler, parallel, N, nchains; kwargs...)
end

# Default implementations of regular and parallel sampling.

"""
    mcmcsample([rng, ]model, sampler, N; kwargs...)

Return `N` samples from the MCMC `sampler` for the provided `model`.

A callback function `f` with type signature
```julia
f(rng, model, sampler, transition, iteration)
```
may be provided as keyword argument `callback`. It is called after every sampling step.
"""
function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    progress = true,
    progressname = "Sampling",
    callback = (args...) -> nothing,
    chain_type::Type=Any,
    kwargs...
)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")

    # Perform any necessary setup.
    sample_init!(rng, model, sampler, N; kwargs...)

    @ifwithprogresslogger progress name=progressname begin
        # Obtain the initial transition.
        transition = step!(rng, model, sampler, N; iteration=1, kwargs...)

        # Run callback.
        callback(rng, model, sampler, transition, 1)

        # Save the transition.
        transitions = AbstractMCMC.transitions(transition, model, sampler, N; kwargs...)
        transitions = save!!(transitions, transition, 1, model, sampler, N; kwargs...)

        # Update the progress bar.
        progress && ProgressLogging.@logprogress 1/N

        # Step through the sampler.
        for i in 2:N
            # Obtain the next transition.
            transition = step!(rng, model, sampler, N, transition; iteration=i, kwargs...)

            # Run callback.
            callback(rng, model, sampler, transition, i)

            # Save the transition.
            transitions = save!!(transitions, transition, i, model, sampler, N; kwargs...)

            # Update the progress bar.
            progress && ProgressLogging.@logprogress i/N
        end
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, model, sampler, N, transitions; kwargs...)

    return bundle_samples(rng, model, sampler, N, transitions, chain_type; kwargs...)
end

"""
    mcmcsample([rng, ]model, sampler, isdone; kwargs...)

Continuously draw samples until a convergence criterion `isdone` returns `true`.

The function `isdone` has the signature
```julia
isdone(rng, model, sampler, transitions, iteration; kwargs...)
```
and should return `true` when sampling should end, and `false` otherwise.

A callback function `f` with type signature
```julia
f(rng, model, sampler, transition, iteration)
```
may be provided as keyword argument `callback`. It is called after every sampling step.
"""
function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    isdone;
    chain_type::Type=Any,
    progress = true,
    progressname = "Convergence sampling",
    callback = (args...) -> nothing,
    kwargs...
)
    # Perform any necessary setup.
    sample_init!(rng, model, sampler, 1; kwargs...)

    @ifwithprogresslogger progress name=progressname begin
        # Obtain the initial transition.
        transition = step!(rng, model, sampler, 1; iteration=1, kwargs...)

        # Run callback.
        callback(rng, model, sampler, transition, 1)

        # Save the transition.
        transitions = AbstractMCMC.transitions(transition, model, sampler; kwargs...)
        transitions = save!!(transitions, transition, 1, model, sampler; kwargs...)

        # Step through the sampler until stopping.
        i = 2

        while !isdone(rng, model, sampler, transitions, i; progress=progress, kwargs...)
            # Obtain the next transition.
            transition = step!(rng, model, sampler, 1, transition; iteration=i, kwargs...)

            # Run callback.
            callback(rng, model, sampler, transition, i)

            # Save the transition.
            transitions = save!!(transitions, transition, i, model, sampler; kwargs...)

            # Increment iteration counter.
            i += 1
        end
    end

    # Wrap up the sampler, if necessary.
    sample_end!(rng, model, sampler, i, transitions; kwargs...)

    # Wrap the samples up.
    return bundle_samples(rng, model, sampler, i, transitions, chain_type; kwargs...)
end

"""
    mcmcsample([rng, ]model, sampler, parallel, N, nchains; kwargs...)

Sample `nchains` chains in parallel using the `parallel` algorithm, and combine them into a
single chain.
"""
function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCThreads,
    N::Integer,
    nchains::Integer;
    progress = true,
    progressname = "Sampling ($(Threads.nthreads()) threads)",
    kwargs...
)
    # Check if actually multiple threads are used.
    if Threads.nthreads() == 1
        @warn "Only a single thread available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Copy the random number generator, model, and sample for each thread
    interval = 1:min(nchains, Threads.nthreads())
    rngs = [deepcopy(rng) for _ in interval]
    models = [deepcopy(model) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @ifwithprogresslogger progress name=progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Distributed.RemoteChannel(() -> Channel{Bool}(nchains))
        end

        Distributed.@sync begin
            if progress
                Distributed.@async begin
                    # Update the progress bar.
                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        ProgressLogging.@logprogress progresschains/nchains
                    end
                end
            end

            Distributed.@async begin
                Threads.@threads for i in 1:nchains
                    # Obtain the ID of the current thread.
                    id = Threads.threadid()

                    # Seed the thread-specific random number generator with the pre-made seed.
                    subrng = rngs[id]
                    Random.seed!(subrng, seeds[i])

                    # Sample a chain and save it to the vector.
                    chains[i] = StatsBase.sample(subrng, models[id], samplers[id], N;
                                                 progress = false, kwargs...)

                    # Update the progress bar.
                    progress && put!(channel, true)
                end

                # Stop updating the progress bar.
                progress && put!(channel, false)
            end
        end
    end

    # Concatenate the chains together.
    return reduce(chainscat, chains)
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCDistributed,
    N::Integer,
    nchains::Integer;
    progress = true,
    progressname = "Sampling ($(Distributed.nworkers()) processes)",
    kwargs...
)
    # Check if actually multiple processes are used.
    if Distributed.nworkers() == 1
        @warn "Only a single process available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up worker pool.
    pool = Distributed.CachingPool(Distributed.workers())

    # Create a channel for progress logging.
    channel = progress ? Distributed.RemoteChannel(() -> Channel{Bool}(nchains)) : nothing

    local chains
    @ifwithprogresslogger progress name=progressname begin
        Distributed.@sync begin
            # Update the progress bar.
            if progress
                Distributed.@async begin
                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        ProgressLogging.@logprogress progresschains/nchains
                    end
                end
            end

            Distributed.@async begin
                chains = let rng=rng, model=model, sampler=sampler, N=N, channel=channel,
                    kwargs=kwargs
                    Distributed.pmap(pool, seeds) do seed
                        # Seed a new random number generator with the pre-made seed.
                        subrng = deepcopy(rng)
                        Random.seed!(subrng, seed)

                        # Sample a chain.
                        chain = StatsBase.sample(subrng, model, sampler, N;
                                                 progress = false, kwargs...)

                        # Update the progress bar.
                        channel === nothing || put!(channel, true)

                        # Return the new chain.
                        return chain
                    end
                end

                # Stop updating the progress bar.
                progress && put!(channel, false)
            end
        end
    end

    # Concatenate the chains together.
    return reduce(chainscat, chains)
end

# Deprecations.
Base.@deprecate psample(model, sampler, N, nchains; kwargs...) sample(model, sampler, MCMCThreads(), N, nchains; kwargs...) false
Base.@deprecate psample(rng, model, sampler, N, nchains; kwargs...) sample(rng, model, sampler, MCMCThreads(), N, nchains; kwargs...) false
Base.@deprecate mcmcpsample(rng, model, sampler, N, nchains; kwargs...) mcmcsample(rng, model, sampler, MCMCThreads(), N, nchains; kwargs...) false
