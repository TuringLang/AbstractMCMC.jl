# Callbacks for AbstractMCMC
# This module provides the unified callback API and supporting types.

###########################
### Core Callback Types ###
###########################

"""
    MultiCallback

A callback that combines multiple callbacks into one.

Implements `push!!` from [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl) to add callbacks to the list.
"""
struct MultiCallback{Cs}
    callbacks::Cs
end

MultiCallback() = MultiCallback(())
MultiCallback(callbacks...) = MultiCallback(callbacks)

(c::MultiCallback)(args...; kwargs...) = foreach(c -> c(args...; kwargs...), c.callbacks)

function BangBang.push!!(c::MultiCallback{<:Tuple}, callback)
    return MultiCallback((c.callbacks..., callback))
end
function BangBang.push!!(c::MultiCallback{<:AbstractArray}, callback)
    (push!(c.callbacks, callback); return c)
end

"""
    NameFilter(; include=nothing, exclude=nothing)

A filter for variable names.

- If `include` is not `nothing`, only names in `include` will pass the filter.
- If `exclude` is not `nothing`, names in `exclude` will be excluded.
"""
Base.@kwdef struct NameFilter{A,B}
    include::A = nothing
    exclude::B = nothing
end

(f::NameFilter)(name, value) = f(name)
function (f::NameFilter)(name)
    include, exclude = f.include, f.exclude
    return (exclude === nothing || name ∉ exclude) &&
           (include === nothing || name ∈ include)
end

##############################
### Defaults and Utilities ###
##############################

const DEFAULT_STATS_OPTIONS = (; thin=0, skip=0, window=typemax(Int))

const DEFAULT_NAME_FILTER = (;
    include=String[], exclude=String[], extras=false, hyperparams=false
)

"""
    merge_with_defaults(user_options::NamedTuple, defaults::NamedTuple)

Merge user-provided options with defaults, where user options take precedence.
"""
function merge_with_defaults(user_options::NamedTuple, defaults::NamedTuple)
    return merge(defaults, user_options)
end
merge_with_defaults(::Nothing, defaults::NamedTuple) = defaults

################################
### Statistics Creation API  ###
################################

"""
    create_stats_with_options(stats, stats_options, num_bins)

Internal constructor for statistics handlers.

If `stats === nothing`, no statistics are collected and `nothing` is returned.
If `stats` is provided, this function requires the OnlineStats extension to be
loaded; otherwise, an error is thrown.

Supports special values:
- `stats=true` or `stats=:default`: Use default statistics (Mean, Variance, KHist)
- `stats=<OnlineStat>`: Use the provided OnlineStat (requires OnlineStats to be loaded)
- `stats=<Tuple of OnlineStats>`: Use multiple stats

This function is not part of the public API and may change or break at any time.
"""
create_stats_with_options(::Nothing, stats_options, num_bins) = nothing

function create_stats_with_options(stats, stats_options, num_bins)
    ext = Base.get_extension(@__MODULE__, :AbstractMCMCOnlineStatsExt)
    if ext === nothing
        error(
            "Statistics collection requires OnlineStats.jl. " *
            "Please load OnlineStats before enabling statistics: `using OnlineStats`",
        )
    end

    # Delegate to OnlineStatsExt for actual creation
    return ext.create_stats_with_options_impl(stats, stats_options, num_bins)
end

################################
### Parameter Extraction API ###
################################

"""
    default_param_names_for_values(x)

Return an iterator of `θ[i]` for each element in `x`.
"""
default_param_names_for_values(x) = ("θ[$i]" for i in 1:length(x))

"""
    _names_and_values(
        model,
        sampler,
        transition,
        state;
        params::Bool = true,
        hyperparams::Bool = false,
        extra::Bool = false,
        kwargs...
    )

Return an iterator over parameter names and values.

This function is not part of the public API and may change or break at any time.

## Keywords
- `params`: include model parameters.
- `hyperparams`: include sampler hyperparameters
- `extra`: include additional statistics.
- `kwargs...`: reserved for internal extensibility.
"""
function _names_and_values(
    model,
    sampler,
    transition,
    state;
    params::Bool=true,
    hyperparams::Bool=false,
    extra::Bool=false,
    kwargs...,
)
    iters = []

    if params
        try
            p = getparams(state)
            push!(iters, zip(default_param_names_for_values(p), p))
        catch
            # No params available
        end
    end

    if hyperparams
        hp = _hyperparams_impl(model, sampler, state; kwargs...)
        if !isempty(hp)
            push!(iters, hp)
        end
    end

    if extra
        try
            stats = getstats(state)
            if stats isa NamedTuple
                push!(iters, pairs(stats))
            end
        catch
            # No extras available
        end
    end

    return Iterators.flatten(iters)
end

# Internal helper for hyperparams extraction
function _hyperparams_impl(model, sampler, state; kwargs...)
    return Pair{String,Any}[]
end

"""
    hyperparam_metrics(model, sampler[, state]; kwargs...)

Return a Vector{String} of metrics for hyperparameters.
Override this to specify which logged values should be used as hyperparam metrics in TensorBoard.
"""
hyperparam_metrics(model, sampler; kwargs...) = String[]
function hyperparam_metrics(model, sampler, state; kwargs...)
    return hyperparam_metrics(model, sampler; kwargs...)
end

#################################
### Unified mcmc_callback API ###
#################################

"""
    mcmc_callback(f::Function)

Create a callback from a function with signature:
`f(rng, model, sampler, transition, state, iteration; kwargs...)`

# Example
```julia
cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
    println("Iteration: \$iteration")
end
```
"""
mcmc_callback(f::Function) = MultiCallback((f,))

"""
    mcmc_callback(callbacks...)

Combine multiple callbacks into one.
"""
mcmc_callback(callbacks::Vararg{Any,N}) where {N} = MultiCallback(callbacks)

"""
    mcmc_callback(;
        logger = nothing,
        stats = nothing,
        stats_options = nothing,
        name_filter = nothing,
    )

Create a callback using keyword arguments.

# Arguments
- `logger`: An `AbstractLogger` instance (e.g., `TBLogger` from TensorBoardLogger.jl)
- `stats`: Statistics to collect. Can be:
  - `nothing`: No statistics (default)
  - `true` or `:default`: Use default statistics (Mean, Variance, KHist) - requires OnlineStats
  - An OnlineStat or tuple of OnlineStats - requires OnlineStats
- `stats_options`: NamedTuple with `thin`, `skip`, `window`
- `name_filter`: NamedTuple with `include`, `exclude`, `extras`, `hyperparams`

# Examples
```julia
# Basic logging (no statistics)
using TensorBoardLogger
lg = TBLogger("runs/exp")
cb = mcmc_callback(logger=lg)

# With default stats (requires OnlineStats)
using TensorBoardLogger, OnlineStats
lg = TBLogger("runs/exp")
cb = mcmc_callback(logger=lg, stats=true)  # or stats=:default

# With custom stats
cb = mcmc_callback(logger=lg, stats=(Mean(), Variance()))
```
"""
function mcmc_callback(;
    logger=nothing,
    stats=nothing,
    stats_options=nothing,
    name_filter=nothing,
    num_bins::Int=100,
)
    # Validate that logger is provided
    if logger === nothing
        throw(
            ArgumentError(
                "A logger must be specified. Pass an AbstractLogger instance to `logger`. " *
                "For TensorBoard logging: `using TensorBoardLogger; mcmc_callback(logger=TBLogger(\"runs/exp\"))`",
            ),
        )
    end

    if !(logger isa Logging.AbstractLogger)
        throw(
            ArgumentError(
                "Expected an AbstractLogger instance, got $(typeof(logger)). " *
                "For TensorBoard: `using TensorBoardLogger; mcmc_callback(logger=TBLogger(\"runs/exp\"))`",
            ),
        )
    end

    merged_stats_options = merge_with_defaults(stats_options, DEFAULT_STATS_OPTIONS)
    merged_name_filter = merge_with_defaults(name_filter, DEFAULT_NAME_FILTER)

    # Process stats in main package - this provides helpful error if OnlineStats not loaded
    processed_stats = create_stats_with_options(stats, merged_stats_options, num_bins)

    callback = _make_logger_callback(
        logger; stats=processed_stats, name_filter=merged_name_filter
    )

    return MultiCallback((callback,))
end

function _make_logger_callback(logger; stats, name_filter)
    ext = Base.get_extension(@__MODULE__, :AbstractMCMCTensorBoardLoggerExt)
    if ext === nothing
        error(
            "TensorBoard logging requires TensorBoardLogger.jl to be loaded. " *
            "Please run: `using TensorBoardLogger`",
        )
    end
    return ext.create_tensorboard_callback(logger; stats, name_filter)
end

"""
    mcmc_callback(existing::MultiCallback, new_callbacks...)

Add callbacks to an existing MultiCallback.
"""
function mcmc_callback(existing::MultiCallback, new_callbacks...)
    result = existing
    for cb in new_callbacks
        result = BangBang.push!!(result, cb)
    end
    return result
end
