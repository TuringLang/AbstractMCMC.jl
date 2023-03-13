# Default implementations of `sample`.
const PROGRESS = Ref(true)

"""
    setprogress!(progress::Bool)

Enable progress logging globally if `progress` is `true`, and disable it otherwise.
"""
function setprogress!(progress::Bool)
    @info "progress logging is $(progress ? "enabled" : "disabled") globally"
    PROGRESS[] = progress
    return progress
end

function StatsBase.sample(
    model_or_logdensity, sampler::AbstractSampler, N_or_isdone; kwargs...
)
    return StatsBase.sample(
        Random.default_rng(), model_or_logdensity, sampler, N_or_isdone; kwargs...
    )
end

"""
    sample(
        rng::Random.AbatractRNG=Random.default_rng(),
        model::AbstractModel,
        sampler::AbstractSampler,
        N_or_isdone;
        kwargs...,
    )

Sample from the `model` with the Markov chain Monte Carlo `sampler` and return the samples.

If `N_or_isdone` is an `Integer`, exactly `N_or_isdone` samples are returned.

Otherwise, sampling is performed until a convergence criterion `N_or_isdone` returns `true`.
The convergence criterion has to be a function with the signature
```julia
isdone(rng, model, sampler, samples, state, iteration; kwargs...)
```
where `state` and `iteration` are the current state and iteration of the sampler, respectively.
It should return `true` when sampling should end, and `false` otherwise.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N_or_isdone;
    kwargs...,
)
    return mcmcsample(rng, model, sampler, N_or_isdone; kwargs...)
end

function StatsBase.sample(
    model_or_logdensity,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...,
)
    return StatsBase.sample(
        Random.default_rng(), model_or_logdensity, sampler, parallel, N, nchains; kwargs...
    )
end

"""
    sample(
        rng::Random.AbstractRNG=Random.default_rng(),
        model::AbstractModel,
        sampler::AbstractSampler,
        parallel::AbstractMCMCEnsemble,
        N::Integer,
        nchains::Integer;
        kwargs...,
    )

Sample `nchains` Monte Carlo Markov chains from the `model` with the `sampler` in parallel
using the `parallel` algorithm, and combine them into a single chain.
"""
function StatsBase.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...,
)
    return mcmcsample(rng, model, sampler, parallel, N, nchains; kwargs...)
end

# Default implementations of regular and parallel sampling.

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    N::Integer;
    progress=PROGRESS[],
    progressname="Sampling",
    callback=nothing,
    discard_initial=0,
    thinning=1,
    chain_type::Type=Any,
    initial_state=nothing,
    kwargs...,
)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    Ntotal = thinning * (N - 1) + discard_initial + 1

    # Start the timer
    start = time()
    local state

    @ifwithprogresslogger progress name = progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        if progress
            threshold = Ntotal ÷ 200
            next_update = threshold
        end

        # Obtain the initial sample and state.
        sample, state = if initial_state === nothing
            step(rng, model, sampler; kwargs...)
        else
            step(rng, model, sampler, state; kwargs...)
        end

        # Discard initial samples.
        for i in 1:discard_initial
            # Update the progress bar.
            if progress && i >= next_update
                ProgressLogging.@logprogress i / Ntotal
                next_update = i + threshold
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler, N; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler, N; kwargs...)

        # Update the progress bar.
        itotal = 1 + discard_initial
        if progress && itotal >= next_update
            ProgressLogging.@logprogress itotal / Ntotal
            next_update = itotal + threshold
        end

        # Step through the sampler.
        for i in 2:N
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = step(rng, model, sampler, state; kwargs...)

                # Update progress bar.
                if progress && (itotal += 1) >= next_update
                    ProgressLogging.@logprogress itotal / Ntotal
                    next_update = itotal + threshold
                end
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)

            # Run callback.
            callback === nothing ||
                callback(rng, model, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler, N; kwargs...)

            # Update the progress bar.
            if progress && (itotal += 1) >= next_update
                ProgressLogging.@logprogress itotal / Ntotal
                next_update = itotal + threshold
            end
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = SamplingStats(start, stop, duration)

    return bundle_samples(
        samples,
        model,
        sampler,
        state,
        chain_type;
        stats=stats,
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...,
    )
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    isdone;
    chain_type::Type=Any,
    progress=PROGRESS[],
    progressname="Convergence sampling",
    callback=nothing,
    discard_initial=0,
    thinning=1,
    initial_state=nothing,
    kwargs...,
)

    # Start the timer
    start = time()
    local state

    @ifwithprogresslogger progress name = progressname begin
        # Obtain the initial sample and state.
        sample, state = if initial_state === nothing
            step(rng, model, sampler; kwargs...)
        else
            step(rng, model, sampler, state; kwargs...)
        end

        # Discard initial samples.
        for _ in 1:discard_initial
            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler; kwargs...)

        # Step through the sampler until stopping.
        i = 2

        while !isdone(rng, model, sampler, samples, state, i; progress=progress, kwargs...)
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = step(rng, model, sampler, state; kwargs...)
            end

            # Obtain the next sample and state.
            sample, state = step(rng, model, sampler, state; kwargs...)

            # Run callback.
            callback === nothing ||
                callback(rng, model, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler; kwargs...)

            # Increment iteration counter.
            i += 1
        end
    end

    # Get the sample stop time.
    stop = time()
    duration = stop - start
    stats = SamplingStats(start, stop, duration)

    # Wrap the samples up.
    return bundle_samples(
        samples,
        model,
        sampler,
        state,
        chain_type;
        stats=stats,
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...,
    )
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCThreads,
    N::Integer,
    nchains::Integer;
    progress=PROGRESS[],
    progressname="Sampling ($(min(nchains, Threads.nthreads())) threads)",
    init_params=nothing,
    initial_state=nothing,
    kwargs...,
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
    nchunks = min(nchains, Threads.nthreads())
    chunksize = cld(nchains, nchunks)
    interval = 1:nchunks
    rngs = [deepcopy(rng) for _ in interval]
    models = [deepcopy(model) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Ensure that initial parameters are `nothing` or of the correct length
    check_initial_params(init_params, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @ifwithprogresslogger progress name = progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Channel{Bool}(length(interval))
        end

        Distributed.@sync begin
            if progress
                # Update the progress bar.
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    threshold = nchains ÷ 200
                    nextprogresschains = threshold

                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        if progresschains >= nextprogresschains
                            ProgressLogging.@logprogress progresschains / nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    Distributed.@sync for (i, _rng, _model, _sampler) in
                                          zip(1:nchunks, rngs, models, samplers)
                        chainidxs = if i == nchunks
                            ((i - 1) * chunksize + 1):nchains
                        else
                            ((i - 1) * chunksize + 1):(i * chunksize)
                        end
                        Threads.@spawn for chainidx in chainidxs
                            # Seed the chunk-specific random number generator with the pre-made seed.
                            Random.seed!(_rng, seeds[chainidx])

                            # Sample a chain and save it to the vector.
                            chains[chainidx] = StatsBase.sample(
                                _rng,
                                _model,
                                _sampler,
                                N;
                                progress=false,
                                init_params=if init_params === nothing
                                    nothing
                                else
                                    init_params[chainidx]
                                end,
                                initial_state=if initial_state === nothing
                                    nothing
                                else
                                    initial_state[chainidx]
                                end
                                kwargs...,
                            )

                            # Update the progress bar.
                            progress && put!(channel, true)
                        end
                    end
                finally
                    # Stop updating the progress bar.
                    progress && put!(channel, false)
                end
            end
        end
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCDistributed,
    N::Integer,
    nchains::Integer;
    progress=PROGRESS[],
    progressname="Sampling ($(Distributed.nworkers()) processes)",
    init_params=nothing,
    initial_state=nothing,
    kwargs...,
)
    # Check if actually multiple processes are used.
    if Distributed.nworkers() == 1
        @warn "Only a single process available: MCMC chains are not sampled in parallel"
    end

    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Ensure that initial parameters are `nothing` or of the correct length
    check_initial_params(init_params, nchains)

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Create initial parameters.
    _init_params = init_params === nothing ? fill(nothing, nchains) : init_params
    _initial_state = initial_state === nothing ? fill(nothing, nchains) : initial_state

    # Set up worker pool.
    pool = Distributed.CachingPool(Distributed.workers())

    local chains
    @ifwithprogresslogger progress name = progressname begin
        # Create a channel for progress logging.
        if progress
            channel = Distributed.RemoteChannel(() -> Channel{Bool}(Distributed.nworkers()))
        end

        Distributed.@sync begin
            if progress
                # Update the progress bar.
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    threshold = nchains ÷ 200
                    nextprogresschains = threshold

                    progresschains = 0
                    while take!(channel)
                        progresschains += 1
                        if progresschains >= nextprogresschains
                            ProgressLogging.@logprogress progresschains / nchains
                            nextprogresschains = progresschains + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    function sample_chain(seed, init_params, initial_state)
                        # Seed a new random number generator with the pre-made seed.
                        Random.seed!(rng, seed)

                        # Sample a chain.
                        chain = StatsBase.sample(
                            rng,
                            model,
                            sampler,
                            N;
                            progress=false,
                            init_params=init_params,
                            initial_state=initial_state,
                            kwargs...,
                        )

                        # Update the progress bar.
                        progress && put!(channel, true)

                        # Return the new chain.
                        return chain
                    end
                    chains = Distributed.pmap(sample_chain, pool, seeds, _init_params, _initial_state)
                finally
                    # Stop updating the progress bar.
                    progress && put!(channel, false)
                end
            end
        end
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

function mcmcsample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    ::MCMCSerial,
    N::Integer,
    nchains::Integer;
    progressname="Sampling",
    init_params=nothing,
    kwargs...,
)
    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Ensure that initial parameters are `nothing` or of the correct length
    check_initial_params(init_params, nchains)

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Sample the chains.
    function sample_chain(i, seed, init_params=nothing)
        # Seed a new random number generator with the pre-made seed.
        Random.seed!(rng, seed)

        # Sample a chain.
        return StatsBase.sample(
            rng,
            model,
            sampler,
            N;
            progressname=string(progressname, " (Chain ", i, " of ", nchains, ")"),
            init_params=init_params,
            kwargs...,
        )
    end

    chains = if init_params === nothing
        map(sample_chain, 1:nchains, seeds)
    else
        map(sample_chain, 1:nchains, seeds, init_params)
    end

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

tighten_eltype(x) = x
tighten_eltype(x::Vector{Any}) = map(identity, x)

@nospecialize check_initial_params(x, n) = throw(
    ArgumentError(
        "initial parameters must be specified as a vector of length equal to the number of chains or `nothing`",
    ),
)

check_initial_params(::Nothing, n) = nothing
function check_initial_params(x::AbstractArray, n)
    if length(x) != n
        throw(
            ArgumentError(
                "incorrect number of initial parameters (expected $n, received $(length(x))"
            ),
        )
    end

    return nothing
end
