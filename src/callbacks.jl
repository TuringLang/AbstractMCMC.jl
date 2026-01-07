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

"""
    Callback{Cs}

A wrapper type that holds a `MultiCallback` internally for type stability.
All callbacks created via `mcmc_callback` are wrapped in this type.
"""
struct Callback{Cs}
    multi::MultiCallback{Cs}
end

(c::Callback)(args...; kwargs...) = c.multi(args...; kwargs...)

function BangBang.push!!(c::Callback, callback)
    new_multi = BangBang.push!!(c.multi, callback)
    return Callback(new_multi)
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
### Parameter Extraction API ###
################################

"""
    default_param_names_for_values(x)

Return an iterator of `θ[i]` for each element in `x`.
"""
default_param_names_for_values(x) = ("θ[$i]" for i in 1:length(x))

"""
    params_and_values(model, state; kwargs...)
    params_and_values(model, sampler, state; kwargs...)
    params_and_values(model, transition, state; kwargs...)
    params_and_values(model, sampler, transition, state; kwargs...)

Return an iterator over parameter names and values.
"""
function params_and_values(model, state; kwargs...)
    try
        params = getparams(state)
        return zip(default_param_names_for_values(params), params)
    catch
        return ()
    end
end

function params_and_values(model, sampler::AbstractSampler, state; kwargs...)
    return params_and_values(model, state; kwargs...)
end

function params_and_values(model, transition, state; kwargs...)
    vals = params_and_values(model, transition; kwargs...)
    return isempty(vals) ? params_and_values(model, state; kwargs...) : vals
end

function params_and_values(model, sampler::AbstractSampler, transition, state; kwargs...)
    return params_and_values(model, transition, state; kwargs...)
end

"""
    extras(model, state; kwargs...)

Return an iterator of (name, value) pairs for additional statistics.
"""
function extras(model, state; kwargs...)
    try
        stats = getstats(state)
        stats isa NamedTuple ? pairs(stats) : ()
    catch
        return ()
    end
end

extras(model, sampler::AbstractSampler, state; kwargs...) = extras(model, state; kwargs...)
extras(model, transition, state; kwargs...) = extras(model, state; kwargs...)
function extras(model, sampler::AbstractSampler, transition, state; kwargs...)
    return extras(model, transition, state; kwargs...)
end

"""
    hyperparams(model, sampler[, state]; kwargs...)

Return an iterator of (name, value) pairs for hyperparameters.
"""
hyperparams(model, sampler; kwargs...) = Pair{String,Any}[]
hyperparams(model, sampler, state; kwargs...) = hyperparams(model, sampler; kwargs...)

"""
    hyperparam_metrics(model, sampler[, state]; kwargs...)

Return a Vector{String} of metrics for hyperparameters.
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
mcmc_callback(f::Function) = Callback(MultiCallback((f,)))

"""
    mcmc_callback(callbacks...)

Combine multiple callbacks into one.
"""
mcmc_callback(callbacks::Vararg{Any,N}) where {N} = Callback(MultiCallback(callbacks))

"""
    mcmc_callback(;
        logger = nothing,
        logdir = nothing,
        stats = nothing,
        stats_options = nothing,
        name_filter = nothing,
    )

Create a callback using keyword arguments.

# Arguments
- `logger`: Logger - can be `:TBLogger` for TensorBoard, or an `AbstractLogger` instance
- `logdir`: Directory for logs (creates TBLogger if logger not specified)
- `stats`: Statistics to collect (OnlineStat or tuple of OnlineStats)
- `stats_options`: NamedTuple with `thin`, `skip`, `window`
- `name_filter`: NamedTuple with `include`, `exclude`, `extras`, `hyperparams`

# Examples
```julia
# With logdir (creates TBLogger automatically)
cb = mcmc_callback(logdir="runs/exp")

# With custom logger
cb = mcmc_callback(logger=my_custom_logger)

# With stats and options
cb = mcmc_callback(logdir="runs/exp", stats=(Mean(), Variance()))
cb = mcmc_callback(logdir="runs/exp", stats_options=(skip=100, thin=5))
```
"""
function mcmc_callback(;
    logger=nothing,
    logdir=nothing,
    stats=nothing,
    stats_options=nothing,
    name_filter=nothing,
)
    # Infer logger type from arguments
    if logger === nothing && logdir !== nothing
        logger = :TBLogger
    end

    # Validate that we have something to create
    if logger === nothing
        throw(
            ArgumentError(
                "At least one callback type must be specified. " *
                "Use `logger=:TBLogger`, `logdir=...`, or pass an AbstractLogger to `logger`.",
            ),
        )
    end

    merged_stats_options = merge_with_defaults(stats_options, DEFAULT_STATS_OPTIONS)
    merged_name_filter = merge_with_defaults(name_filter, DEFAULT_NAME_FILTER)

    # Create callback based on logger type
    callbacks = if logger === :TBLogger
        (
            _make_tensorboard_callback(
                logdir;
                logger=nothing,
                stats,
                stats_options=merged_stats_options,
                name_filter=merged_name_filter,
            ),
        )
    elseif logger isa Logging.AbstractLogger
        # Custom logger instance passed directly
        (
            _make_tensorboard_callback(
                nothing;
                logger,
                stats,
                stats_options=merged_stats_options,
                name_filter=merged_name_filter,
            ),
        )
    else
        throw(
            ArgumentError(
                "Unknown logger type: $(typeof(logger)). Use :TBLogger or pass an AbstractLogger.",
            ),
        )
    end

    return Callback(MultiCallback(callbacks))
end

function _make_tensorboard_callback(logdir; logger, stats, stats_options, name_filter)
    ext = Base.get_extension(@__MODULE__, :AbstractMCMCTensorBoardLoggerExt)
    if ext === nothing
        error(
            "TensorBoard logging requires both TensorBoardLogger and OnlineStats. " *
            "Please run: `using TensorBoardLogger, OnlineStats`",
        )
    end
    return ext.create_tensorboard_callback(
        logdir; logger, stats, stats_options, name_filter
    )
end

"""
    mcmc_callback(existing::Callback, new_callbacks...)

Add callbacks to an existing Callback.
"""
function mcmc_callback(existing::Callback, new_callbacks...)
    result = existing
    for cb in new_callbacks
        result = BangBang.push!!(result, cb)
    end
    return result
end
