# Callbacks for AbstractMCMC
# This module provides the unified callback API and supporting types.

###########################
### Core Callback Types ###
###########################

"""
    MultiCallback

A callback that combines multiple callbacks into one.

Supports `push!!` from [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl) to add callbacks,
returning a new `MultiCallback` with the added callback.
"""
struct MultiCallback{Cs<:Tuple}
    callbacks::Cs
end

MultiCallback() = MultiCallback(())
MultiCallback(callbacks...) = MultiCallback(callbacks)

(c::MultiCallback)(args...; kwargs...) = foreach(c -> c(args...; kwargs...), c.callbacks)

function BangBang.push!!(c::MultiCallback, callback)
    return MultiCallback((c.callbacks..., callback))
end

"""
    NameFilter(; include=Set{String}(), exclude=Set{String}())

A filter for variable names.

- If `include` is non-empty, only names in `include` will pass the filter.
- Names in `exclude` will be excluded.
- Throws an error if `include` and `exclude` have overlapping elements.
"""
struct NameFilter
    include::Set{String}
    exclude::Set{String}

    function NameFilter(; include=Set{String}(), exclude=Set{String}())
        inc_set = include isa Set ? include : Set{String}(include)
        exc_set = exclude isa Set ? exclude : Set{String}(exclude)
        overlap = intersect(inc_set, exc_set)
        if !isempty(overlap)
            error("NameFilter: include and exclude have overlapping elements: $overlap")
        end
        return new(inc_set, exc_set)
    end
end

(f::NameFilter)(name, value) = f(name)
function (f::NameFilter)(name)
    return name ∉ f.exclude && (isempty(f.include) || name ∈ f.include)
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

Combine multiple callbacks into one. Requires at least one callback.
"""
function mcmc_callback(cb1, callbacks...)
    return MultiCallback((cb1, callbacks...))
end

"""
    mcmc_callback(;
        logger,
        stats = nothing,
        stats_options = nothing,
        name_filter = nothing,
    )

Create a TensorBoard logging callback. **Requires TensorBoardLogger.jl to be loaded.**

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
using TensorBoardLogger
lg = TBLogger("runs/exp")
cb = mcmc_callback(logger=lg)

# With default stats (requires OnlineStats)
using TensorBoardLogger, OnlineStats
lg = TBLogger("runs/exp")
cb = mcmc_callback(logger=lg, stats=true)
```

!!! note
    This method is defined in the TensorBoardLogger extension. You must load
    TensorBoardLogger before using it: `using TensorBoardLogger`
"""
function mcmc_callback end

"""
    mcmc_callback(existing::MultiCallback, new_callbacks...)

Add callbacks to an existing MultiCallback.
"""
function mcmc_callback(existing::MultiCallback, new_callbacks...)
    return MultiCallback((existing.callbacks..., new_callbacks...))
end
