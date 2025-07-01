# Default implementations of `sample`.
const PROGRESS = Ref(true)
const MAX_CHAINS_PROGRESS = Ref(10)

_pluralise(n; singular="", plural="s") = n == 1 ? singular : plural

"""
    setprogress!(progress::Bool; silent::Bool=false)

Enable progress logging globally if `progress` is `true`, and disable it otherwise. 
Optionally disable informational message if `silent` is `true`.
"""
function setprogress!(progress::Bool; silent::Bool=false)
    if !silent
        @info "progress logging is $(progress ? "enabled" : "disabled") globally"
    end
    PROGRESS[] = progress
    return progress
end

"""
    setmaxchainsprogress!(max_chains::Int, silent::Bool=false)

Set the maximum number of chains to display progress bars for when sampling
multiple chains at once (if progress logging is enabled). Above this limit, no
progress bars are displayed for individual chains; instead, a single progress
bar is displayed for the entire sampling process.
"""
function setmaxchainsprogress!(max_chains::Int, silent::Bool=false)
    if max_chains < 0
        throw(ArgumentError("maximum number of chains must be non-negative"))
    end
    if !silent
        @info "AbstractMCMC: maximum number of per-chain progress bars set to $max_chains"
    end
    MAX_CHAINS_PROGRESS[] = max_chains
    return nothing
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

# Keyword arguments

See https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments for common keyword
arguments.
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

# Keyword arguments

See https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments for common keyword
arguments.
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
    progress::Union{Bool,UUIDs.UUID,Channel{Bool},Distributed.RemoteChannel{Channel{Bool}}}=PROGRESS[],
    progressname="Sampling",
    callback=nothing,
    num_warmup::Int=0,
    discard_initial::Int=num_warmup,
    thinning=1,
    chain_type::Type=Any,
    initial_state=nothing,
    kwargs...,
)
    # Check the number of requested samples.
    N > 0 || error("the number of samples must be ≥ 1")
    discard_initial >= 0 ||
        throw(ArgumentError("number of discarded samples must be non-negative"))
    num_warmup >= 0 ||
        throw(ArgumentError("number of warm-up samples must be non-negative"))
    Ntotal = thinning * (N - 1) + discard_initial + 1
    Ntotal >= num_warmup || throw(
        ArgumentError("number of warm-up samples exceeds the total number of samples")
    )

    # Determine how many samples to drop from `num_warmup` and the
    # main sampling process before we start saving samples.
    discard_from_warmup = min(num_warmup, discard_initial)
    keep_from_warmup = num_warmup - discard_from_warmup

    # Start the timer
    start = time()
    local state

    # Only create a new progress bar if progress is explicitly equal to true, i.e.
    # it's not a UUID (the progress bar already exists), a channel (there's no need
    # for a new progress bar), or false (no progress bar).
    @ifwithprogresslogger (progress == true) name = progressname begin
        # Determine threshold values for progress logging
        # (one update per 0.5% of progress)
        threshold = Ntotal ÷ 200
        next_update = threshold

        # Slightly hacky code to reset the start timer if called from a
        # multi-chain sampling process. We need this because the progress bar
        # is constructed in the multi-chain method, i.e. if we don't do this
        # the progress bar shows the time elapsed since _all_ sampling began,
        # not since the current chain started.
        if progress isa UUIDs.UUID
            try
                bartrees = Logging.current_logger().loggers[1].logger.bartrees
                bar = TerminalLoggers.findbar(bartrees, progress).data
                bar.tfirst = time()
            catch
            end
        end

        # Obtain the initial sample and state.
        sample, state = if num_warmup > 0
            if initial_state === nothing
                step_warmup(rng, model, sampler; kwargs...)
            else
                step_warmup(rng, model, sampler, initial_state; kwargs...)
            end
        else
            if initial_state === nothing
                step(rng, model, sampler; kwargs...)
            else
                step(rng, model, sampler, initial_state; kwargs...)
            end
        end

        # Start the progress bar.
        itotal = 1
        if itotal >= next_update
            @log_progress_dispatch progress progressname itotal / Ntotal
            next_update = itotal + threshold
        end
        if progress isa Channel{Bool} ||
            progress isa Distributed.RemoteChannel{Channel{Bool}}
            put!(progress, true)
        end

        # Discard initial samples.
        for j in 1:discard_initial
            # Obtain the next sample and state.
            sample, state = if j ≤ num_warmup
                step_warmup(rng, model, sampler, state; kwargs...)
            else
                step(rng, model, sampler, state; kwargs...)
            end

            # Update the progress bar.
            itotal += 1
            if itotal >= next_update
                @log_progress_dispatch progress progressname itotal / Ntotal
                next_update = itotal + threshold
            end
        end

        # Run callback.
        callback === nothing || callback(rng, model, sampler, sample, state, 1; kwargs...)

        # Save the sample.
        samples = AbstractMCMC.samples(sample, model, sampler, N; kwargs...)
        samples = save!!(samples, sample, 1, model, sampler, N; kwargs...)

        # Step through the sampler.
        for i in 2:N
            # Discard thinned samples.
            for _ in 1:(thinning - 1)
                # Obtain the next sample and state.
                sample, state = if i ≤ keep_from_warmup
                    step_warmup(rng, model, sampler, state; kwargs...)
                else
                    step(rng, model, sampler, state; kwargs...)
                end

                # Update progress bar.
                itotal += 1
                if itotal >= next_update
                    @log_progress_dispatch progress progressname itotal / Ntotal
                    next_update = itotal + threshold
                end
            end

            # Obtain the next sample and state.
            sample, state = if i ≤ keep_from_warmup
                step_warmup(rng, model, sampler, state; kwargs...)
            else
                step(rng, model, sampler, state; kwargs...)
            end

            # Run callback.
            callback === nothing ||
                callback(rng, model, sampler, sample, state, i; kwargs...)

            # Save the sample.
            samples = save!!(samples, sample, i, model, sampler, N; kwargs...)

            # Update the progress bar.
            itotal += 1
            if itotal >= next_update
                @log_progress_dispatch progress progressname itotal / Ntotal
                next_update = itotal + threshold
            end
            if progress isa Channel{Bool} ||
                progress isa Distributed.RemoteChannel{Channel{Bool}}
                put!(progress, true)
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
    progress::Bool=PROGRESS[],
    progressname="Convergence sampling",
    callback=nothing,
    num_warmup=0,
    discard_initial=num_warmup,
    thinning=1,
    initial_state=nothing,
    kwargs...,
)
    # Check the number of requested samples.
    discard_initial >= 0 ||
        throw(ArgumentError("number of discarded samples must be non-negative"))
    num_warmup >= 0 ||
        throw(ArgumentError("number of warm-up samples must be non-negative"))

    # Determine how many samples to drop from `num_warmup` and the
    # main sampling process before we start saving samples.
    discard_from_warmup = min(num_warmup, discard_initial)
    keep_from_warmup = num_warmup - discard_from_warmup

    # Start the timer
    start = time()
    local state

    @ifwithprogresslogger progress name = progressname begin
        # Obtain the initial sample and state.
        sample, state = if num_warmup > 0
            if initial_state === nothing
                step_warmup(rng, model, sampler; kwargs...)
            else
                step_warmup(rng, model, sampler, initial_state; kwargs...)
            end
        else
            if initial_state === nothing
                step(rng, model, sampler; kwargs...)
            else
                step(rng, model, sampler, initial_state; kwargs...)
            end
        end

        # Discard initial samples.
        for j in 1:discard_initial
            # Obtain the next sample and state.
            sample, state = if j ≤ num_warmup
                step_warmup(rng, model, sampler, state; kwargs...)
            else
                step(rng, model, sampler, state; kwargs...)
            end
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
                sample, state = if i ≤ keep_from_warmup
                    step_warmup(rng, model, sampler, state; kwargs...)
                else
                    step(rng, model, sampler, state; kwargs...)
                end
            end

            # Obtain the next sample and state.
            sample, state = if i ≤ keep_from_warmup
                step_warmup(rng, model, sampler, state; kwargs...)
            else
                step(rng, model, sampler, state; kwargs...)
            end

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
    progress::Union{Bool,Symbol}=PROGRESS[],
    progressname="Sampling ($(min(nchains, Threads.nthreads())) thread$(_pluralise(min(nchains, Threads.nthreads()))))",
    initial_params=nothing,
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

    # Determine default progress bar style.
    if progress == true
        progress = nchains > MAX_CHAINS_PROGRESS[] ? :overall : :perchain
    elseif progress == false
        progress = :none
    end
    progress in [:overall, :perchain, :none] || throw(
        ArgumentError(
            "`progress` for MCMCThreads must be `:overall`, `:perchain`, `:none`, or a boolean",
        ),
    )

    # Copy the random number generator, model, and sample for each thread
    nchunks = min(nchains, Threads.nthreads())
    interval = 1:nchunks
    # `copy` instead of `deepcopy` for RNGs: https://github.com/JuliaLang/julia/issues/42899
    rngs = [copy(rng) for _ in interval]
    models = [deepcopy(model) for _ in interval]
    samplers = [deepcopy(sampler) for _ in interval]

    # If nchains/nchunks = m with remainder n, then the first n chunks will
    # have m + 1 chains, and the rest will have m chains.
    m, n = divrem(nchains, nchunks)

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Ensure that initial parameters and states are `nothing` or of the correct length
    check_initial_params(initial_params, nchains)
    check_initial_state(initial_state, nchains)

    # Set up a chains vector.
    chains = Vector{Any}(undef, nchains)

    @ifwithprogresslogger (progress != :none) name = progressname begin
        if progress == :perchain
            # This is the 'overall' progress bar. We create a channel for each
            # chain to report back to when it finishes sampling.
            progress_channel = Channel{Bool}()
            # These are the per-chain progress bars. We generate `nchains`
            # independent UUIDs for each progress bar
            uuids = [UUIDs.uuid4() for _ in 1:nchains]
            progress_names = ["Chain $i/$nchains" for i in 1:nchains]
            # Start the per-chain progress bars (but in reverse order, because
            # ProgressLogging prints from the bottom up, and we want chain 1 to
            # show up at the top)
            for (progress_name, uuid) in reverse(collect(zip(progress_names, uuids)))
                ProgressLogging.@logprogress name = progress_name nothing _id = uuid
            end
        elseif progress == :overall
            # Just a single progress bar for the entire sampling, but instead
            # of tracking each chain as it comes in, we track each sample as it
            # comes in. This allows us to have more granular progress updates.
            progress_channel = Channel{Bool}(nchains)
        end

        Distributed.@sync begin
            if progress != :none
                # This task updates the progress bar
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    Ntotal = progress == :overall ? nchains * N : nchains
                    threshold = Ntotal ÷ 200
                    next_update = threshold

                    itotal = 0
                    while take!(progress_channel)
                        itotal += 1
                        if itotal >= next_update
                            ProgressLogging.@logprogress itotal / Ntotal
                            next_update = itotal + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    Distributed.@sync for (i, _rng, _model, _sampler) in
                                          zip(interval, rngs, models, samplers)
                        if i <= n
                            chainidx_hi = i * (m + 1)
                            nchains_chunk = m + 1
                        else
                            chainidx_hi = i * m + n # n * (m + 1) + (i - n) * m
                            nchains_chunk = m
                        end
                        chainidx_lo = chainidx_hi - nchains_chunk + 1
                        chainidxs = chainidx_lo:chainidx_hi

                        Threads.@spawn for chainidx in chainidxs
                            # Seed the chunk-specific random number generator with the pre-made seed.
                            Random.seed!(_rng, seeds[chainidx])

                            # Determine how to monitor progress for the child chains.
                            child_progress, child_progressname = if progress == :none
                                # No need to create a progress bar
                                false, ""
                            elseif progress == :overall
                                # No need to create a new progress bar, but we need to
                                # pass the channel to the child so that it can log when
                                # it has finished obtaining each sample.
                                progress_channel, ""
                            elseif progress == :perchain
                                # We need to specify both the ID of the progress bar for
                                # the child to update, and we also specify the name to use
                                # for the progress bar.
                                uuids[chainidx], progress_names[chainidx]
                            end

                            # Sample a chain and save it to the vector.
                            chains[chainidx] = StatsBase.sample(
                                _rng,
                                _model,
                                _sampler,
                                N;
                                progress=child_progress,
                                progressname=child_progressname,
                                initial_params=if initial_params === nothing
                                    nothing
                                else
                                    initial_params[chainidx]
                                end,
                                initial_state=if initial_state === nothing
                                    nothing
                                else
                                    initial_state[chainidx]
                                end,
                                kwargs...,
                            )

                            # Update the progress bars.
                            if progress == :perchain
                                # Tell the 'main' progress bar that this chain is done.
                                put!(progress_channel, true)
                                # Conclude the per-chain progress bar.
                                ProgressLogging.@logprogress progress_names[chainidx] "done" _id = uuids[chainidx]
                            end
                            # Note that if progress == :overall, we don't need to do anything
                            # because progress on that bar is triggered by
                            # samples being obtained rather than chains being
                            # completed.
                        end
                    end
                finally
                    if progress == :perchain
                        # Stop updating the main progress bar (either if sampling
                        # is done, or if an error occurs).
                        put!(progress_channel, false)
                        # Additionally stop the per-chain progress bars
                        for (progress_name, uuid) in zip(progress_names, uuids)
                            ProgressLogging.@logprogress progress_name "done" _id = uuid
                        end
                    elseif progress == :overall
                        # Stop updating the main progress bar (either if sampling
                        # is done, or if an error occurs).
                        put!(progress_channel, false)
                    end
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
    progress::Union{Bool,Symbol}=PROGRESS[],
    progressname="Sampling ($(Distributed.nworkers()) process$(_pluralise(Distributed.nworkers(); plural="es")))",
    initial_params=nothing,
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

    # Determine default progress bar style. Note that for MCMCDistributed(),
    # :perchain isn't implemented.
    if progress == true
        progress = :overall
    elseif progress == false
        progress = :none
    end
    progress in [:overall, :none] || throw(
        ArgumentError(
            "`progress` for MCMCDistributed must be `:overall`, `:none`, or a boolean"
        ),
    )

    # Ensure that initial parameters and states are `nothing` or of the correct length
    check_initial_params(initial_params, nchains)
    check_initial_state(initial_state, nchains)

    _initial_params =
        initial_params === nothing ? FillArrays.Fill(nothing, nchains) : initial_params
    _initial_state =
        initial_state === nothing ? FillArrays.Fill(nothing, nchains) : initial_state

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Set up worker pool.
    pool = Distributed.CachingPool(Distributed.workers())

    local chains
    @ifwithprogresslogger (progress != :none) name = progressname begin
        # Set up progress logging.
        if progress == :overall
            # Just a single progress bar for the entire sampling, but instead
            # of tracking each chain as it comes in, we track each sample as it
            # comes in. This allows us to have more granular progress updates.
            chan = Channel{Bool}(Distributed.nworkers())
            progress_channel = Distributed.RemoteChannel(() -> chan)
            child_progresses = [progress_channel for _ in 1:nchains]
            child_progressnames = ["" for _ in 1:nchains]
        elseif progress == :none
            child_progresses = [false for _ in 1:nchains]
            child_progressnames = ["" for _ in 1:nchains]
        end

        Distributed.@sync begin
            if progress == :overall
                # This task updates the progress bar
                Distributed.@async begin
                    # Determine threshold values for progress logging
                    # (one update per 0.5% of progress)
                    Ntotal = nchains * N
                    threshold = Ntotal ÷ 200
                    next_update = threshold

                    itotal = 0
                    while take!(progress_channel)
                        itotal += 1
                        if itotal >= next_update
                            ProgressLogging.@logprogress itotal / Ntotal
                            next_update = itotal + threshold
                        end
                    end
                end
            end

            Distributed.@async begin
                try
                    function sample_chain(
                        seed,
                        initial_params,
                        initial_state,
                        child_progress,
                        child_progressname,
                    )
                        # Seed a new random number generator with the pre-made seed.
                        Random.seed!(rng, seed)

                        # Sample a chain.
                        chain = StatsBase.sample(
                            rng,
                            model,
                            sampler,
                            N;
                            progress=child_progress,
                            progressname=child_progressname,
                            initial_params=initial_params,
                            initial_state=initial_state,
                            kwargs...,
                        )

                        # Update the progress bars. Note that the case of
                        # progress = :overall doesn't need to be handled here
                        # (for similar reasons to the MCMCThreads method
                        # above).
                        if progress == :perchain
                            # Tell the 'main' progress bar that this chain is done.
                            put!(progress_channel, true)
                            # Conclude the per-chain progress bar.
                            ProgressLogging.@logprogress child_progressname "done" _id =
                                child_progress
                        end

                        # Return the new chain.
                        return chain
                    end
                    chains = Distributed.pmap(
                        sample_chain,
                        pool,
                        seeds,
                        _initial_params,
                        _initial_state,
                        child_progresses,
                        child_progressnames,
                    )
                finally
                    if progress == :overall
                        # Stop updating the main progress bar (either if sampling
                        # is done, or if an error occurs).
                        put!(progress_channel, false)
                    end
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
    initial_params=nothing,
    initial_state=nothing,
    kwargs...,
)
    # Check if the number of chains is larger than the number of samples
    if nchains > N
        @warn "Number of chains ($nchains) is greater than number of samples per chain ($N)"
    end

    # Ensure that initial parameters and states are `nothing` or of the correct length
    check_initial_params(initial_params, nchains)
    check_initial_state(initial_state, nchains)

    _initial_params =
        initial_params === nothing ? FillArrays.Fill(nothing, nchains) : initial_params
    _initial_state =
        initial_state === nothing ? FillArrays.Fill(nothing, nchains) : initial_state

    # Create a seed for each chain using the provided random number generator.
    seeds = rand(rng, UInt, nchains)

    # Sample the chains.
    function sample_chain(i, seed, initial_params, initial_state)
        # Seed a new random number generator with the pre-made seed.
        Random.seed!(rng, seed)

        # Sample a chain.
        return StatsBase.sample(
            rng,
            model,
            sampler,
            N;
            progressname=string(progressname, " (Chain ", i, " of ", nchains, ")"),
            initial_params=initial_params,
            initial_state=initial_state,
            kwargs...,
        )
    end

    chains = map(sample_chain, 1:nchains, seeds, _initial_params, _initial_state)

    # Concatenate the chains together.
    return chainsstack(tighten_eltype(chains))
end

tighten_eltype(x) = x
tighten_eltype(x::Vector{Any}) = map(identity, x)

@nospecialize check_initial_params(x, n) = throw(
    ArgumentError(
        "initial_params must be specified as a vector of length equal to the number of chains or `nothing`",
    ),
)
check_initial_params(::Nothing, n) = nothing
function check_initial_params(x::AbstractArray, n)
    if length(x) != n
        throw(
            ArgumentError(
                "the length of initial_params ($(length(x))) does not equal the number of chains ($n)",
            ),
        )
    end

    return nothing
end

@nospecialize check_initial_state(x, n) = throw(
    ArgumentError(
        "initial_state must be specified as a vector of length equal to the number of chains or `nothing`",
    ),
)
check_initial_state(::Nothing, n) = nothing
function check_initial_state(x::AbstractArray, n)
    if length(x) != n
        throw(
            ArgumentError(
                "the length of initial_state ($(length(x))) does not equal the number of chains ($n)",
            ),
        )
    end

    return nothing
end
