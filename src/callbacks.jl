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
    include=String[], exclude=String[], stats=false, extras=false
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
    ParamsWithStats{P,S,E}

A container for MCMC parameters, statistics, and extras.

This is the **public API** for extracting named values from MCMC states.
Use `Base.pairs(pws)` to iterate over `(name, value)` pairs.

# Fields
- `params::P`: Parameter values (Vector, Dict, or OrderedDict)
- `stats::S`: Statistics as a NamedTuple (e.g., `(lp=...,)`)
- `extras::E`: Extra diagnostics as a NamedTuple

# Example
```julia
pws = ParamsWithStats(model, sampler, transition, state; params=true, stats=true)
for (name, value) in Base.pairs(pws)
    println("\$name: \$value")
end

# Re-select to exclude stats:
pws2 = ParamsWithStats(pws; params=true, stats=false)
```
"""
struct ParamsWithStats{P,S,E}
    params::P
    stats::S
    extras::E
end

"""
    ParamsWithStats(model, sampler, transition, state; params=true, stats=false, extras=false)

Construct a `ParamsWithStats` by extracting values from the MCMC state.

- `params=true`: Include model parameters via `getparams(state)`.
- `stats=true`: Include step-level statistics via `getstats(state)`.
- `extras=true`: Include constant or iteration-level metadata (e.g. hyperparams).
"""
function ParamsWithStats(
    model,
    sampler,
    transition,
    state;
    params::Bool=true,
    stats::Bool=false,
    extras::Bool=false,
)
    p = params ? getparams(state) : nothing
    s = stats ? getstats(state) : NamedTuple()
    e = extras ? NamedTuple() : NamedTuple()  # Samplers can override for actual extras
    return ParamsWithStats(p, s, e)
end

"""
    ParamsWithStats(pws::ParamsWithStats; params=true, stats=true, extras=true)

Create a new `ParamsWithStats` by selecting subsets of an existing one.

This enables filtering without re-extracting from state:
```julia
pws = ParamsWithStats(model, sampler, transition, state; params=true, stats=true)
pws_params_only = ParamsWithStats(pws; params=true, stats=false, extras=false)
```
"""
function ParamsWithStats(
    pws::ParamsWithStats; params::Bool=true, stats::Bool=true, extras::Bool=true
)
    p = params ? pws.params : nothing
    s = stats ? pws.stats : NamedTuple()
    e = extras ? pws.extras : NamedTuple()
    return ParamsWithStats(p, s, e)
end

"""
    Base.pairs(pws::ParamsWithStats)

Return an iterator of `(name, value)` pairs for all selected data in `pws`.

This is the canonical way to iterate over a `ParamsWithStats`:
```julia
for (name, value) in Base.pairs(pws)
    @info name value
end
```
"""
function Base.pairs(pws::ParamsWithStats)
    iters = []

    # Handle params
    if pws.params !== nothing && !isempty(pws.params)
        if first(pws.params) isa Pair
            # Already named pairs - use directly
            push!(iters, pws.params)
        else
            # Raw values - add default θ[i] names
            push!(iters, (n => v for (n, v) in zip(default_param_names_for_values(pws.params), pws.params)))
        end
    end

    # Handle stats
    if !isempty(pws.stats)
        push!(iters, (string(k) => v for (k, v) in Base.pairs(pws.stats)))
    end

    # Handle extras
    if !isempty(pws.extras)
        push!(iters, (string(k) => v for (k, v) in Base.pairs(pws.extras)))
    end

    return Iterators.flatten(iters)
end

function Base.isempty(pws::ParamsWithStats)
    return (
        (pws.params === nothing || isempty(pws.params)) &&
        isempty(pws.stats) &&
        isempty(pws.extras)
    )
end

#################################
### Unified mcmc_callback API ###
#################################

"""
    mcmc_callback(callback)
    mcmc_callback(callbacks...)

Create a callback or combine multiple callbacks into one.

Any callable (function or callable struct) with the signature
`(rng, model, sampler, transition, state, iteration; kwargs...)` can be used.

# Example
```julia
cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
    println("Iteration: \$iteration")
end
```
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
- `name_filter`: NamedTuple with `include`, `exclude`, `stats`, `hyperparams`

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
