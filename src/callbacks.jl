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
    str_name = string(name)
    return str_name ∉ f.exclude && (isempty(f.include) || str_name ∈ f.include)
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
    ParamsWithStats{P,S,E}

A container for MCMC parameters, statistics, and extras. The parameter container can be a
structured type; statistics and extras are stored as `NamedTuple`s. Use `Base.pairs(pws)`
to iterate over all `(name, value)` pairs.

The parameter container must implement `pairs` and `isempty`. For `==`, `isequal`, and
`hash` of `ParamsWithStats` to be meaningful it must also implement those (with `==`
returning `Bool` or `missing`), and its keys should have a meaningful `string` form so
that name-based filtering and logging callbacks work. Keys are not required to be
`Symbol`s, so `pairs(pws)` may yield pairs with mixed key types; consumers should not
assume `Symbol` keys or a concrete element type.

Note that `AbstractVector{<:Real}` and `AbstractVector{<:Pair}` parameter inputs are not
stored as given: the extraction constructors normalize them to `Symbol`-keyed
`NamedTuple`s (see the constructor docs below).

# Fields
- `params::P`: Parameter values in a container implementing `pairs` and `isempty`
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
struct ParamsWithStats{P,S<:NamedTuple,E<:NamedTuple}
    params::P
    stats::S
    extras::E
end

"""
    ParamsWithStats(params, stats::NamedTuple)

Construct a `ParamsWithStats` with no extra diagnostics.
"""
function ParamsWithStats(params, stats::NamedTuple)
    return ParamsWithStats(params, stats, NamedTuple())
end

# Constructor from Vector{<:Real} - adds default θ[i] names
function ParamsWithStats(
    v::AbstractVector{<:Real}, stats::S, extras::E
) where {S<:NamedTuple,E<:NamedTuple}
    names = ntuple(i -> Symbol("θ[$i]"), length(v))
    params = NamedTuple{names}(Tuple(v))
    return ParamsWithStats(params, stats, extras)
end

# Constructor from Vector{Pair} - converts to NamedTuple
function ParamsWithStats(
    v::AbstractVector{<:Pair}, stats::S, extras::E
) where {S<:NamedTuple,E<:NamedTuple}
    names = Tuple(Symbol(first(p)) for p in v)
    values = Tuple(last(p) for p in v)
    params = NamedTuple{names}(values)
    return ParamsWithStats(params, stats, extras)
end

# Constructor for nothing params (when params=false)
function ParamsWithStats(::Nothing, stats::S, extras::E) where {S<:NamedTuple,E<:NamedTuple}
    return ParamsWithStats(NamedTuple(), stats, extras)
end

"""
    ParamsWithStats(model, sampler, transition, state; params=true, stats=false, extras=false)

Construct a `ParamsWithStats` by extracting values from the MCMC state.

# Arguments
- `params=true`: Include model parameters via `getparams(state)`.
- `stats=true`: Include step-level statistics via `getstats(state)`. These are values that
  change once per MCMC iteration (e.g., log probability, acceptance rate).
- `extras=true`: Include extra diagnostics. These are values that remain constant across
  MCMC iterations (e.g., preconditioning matrix, number of particles) or change multiple
  times within a single iteration (e.g., leapfrog phase points in HMC).
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
    e = extras ? NamedTuple() : NamedTuple()
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
    p = params ? pws.params : NamedTuple()
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
    return Iterators.flatten((pairs(pws.params), pairs(pws.stats), pairs(pws.extras)))
end

function Base.isempty(pws::ParamsWithStats)
    return (isempty(pws.params) && isempty(pws.stats) && isempty(pws.extras))
end

function Base.:(==)(pws1::ParamsWithStats, pws2::ParamsWithStats)
    return (pws1.params == pws2.params) & (pws1.stats == pws2.stats) &
           (pws1.extras == pws2.extras)
end

function Base.isequal(pws1::ParamsWithStats, pws2::ParamsWithStats)
    return isequal(pws1.params, pws2.params) &&
           isequal(pws1.stats, pws2.stats) &&
           isequal(pws1.extras, pws2.extras)
end

function Base.hash(pws::ParamsWithStats, h::UInt)
    return hash(pws.extras, hash(pws.stats, hash(pws.params, hash(:ParamsWithStats, h))))
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
- `name_filter`: NamedTuple with `include`, `exclude`, `stats`, `extras`

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
