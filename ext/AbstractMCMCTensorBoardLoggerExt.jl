module AbstractMCMCTensorBoardLoggerExt

using AbstractMCMC
using AbstractMCMC:
    NameFilter,
    params_and_values,
    extras,
    hyperparams,
    hyperparam_metrics,
    Skip,
    Thin,
    WindowStat
using TensorBoardLogger
using TensorBoardLogger: TBLogger
using OnlineStats
using OnlineStats: OnlineStat, Mean, Variance, KHist, Series, AutoCov, fit!, value, nobs
using Logging: AbstractLogger, with_logger, @info, Info
using Dates: now, DateFormat, format

const TBL = TensorBoardLogger

##########################################
### OnlineStats Interface for Wrappers ###
##########################################

# Skip
OnlineStats.nobs(o::Skip) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Skip) = OnlineStats.value(o.stat)

function OnlineStats.fit!(o::Skip, x)
    if o.n >= o.b
        OnlineStats.fit!(o.stat, x)
    end
    o.n += 1
    return o
end

function Base.show(io::IO, o::Skip)
    return print(io, "Skip ($(o.b)): n=$(o.n) | stat=$(o.stat)")
end

# Thin
OnlineStats.nobs(o::Thin) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Thin) = OnlineStats.value(o.stat)

function OnlineStats.fit!(o::Thin, x)
    if (o.n % o.b) == 0
        OnlineStats.fit!(o.stat, x)
    end
    o.n += 1
    return o
end

function Base.show(io::IO, o::Thin)
    return print(io, "Thin ($(o.b)): n=$(o.n) | stat=$(o.stat)")
end

# WindowStat
OnlineStats.nobs(o::WindowStat) = min(o.n, o.b)

function OnlineStats.fit!(o::WindowStat, x)
    o.n += 1
    # Circular buffer logic: insert at the correct position based on total count
    idx = mod1(o.n, o.b)
    o.buffer[idx] = x
    return o
end

function OnlineStats.value(o::WindowStat)
    # Re-fit a clean stat on the current window buffer
    stat_new = deepcopy(o.stat)

    if o.n < o.b
        # Buffer not full yet, data is 1..n
        for i in 1:(o.n)
            fit!(stat_new, o.buffer[i])
        end
    else
        # Buffer is full (circular).
        # We must fit in chronological order: from (n - b + 1) to n.
        for i in (o.n - o.b + 1):(o.n)
            idx = mod1(i, o.b)
            fit!(stat_new, o.buffer[idx])
        end
    end
    return stat_new
end

function Base.show(io::IO, o::WindowStat)
    return print(io, "WindowStat ($(o.b)): n=$(o.n) | stat=$(o.stat)")
end

#########################################
### TensorBoardLogger name formatting ###
#########################################

tb_name(arg) = string(arg)
tb_name(stat::OnlineStat) = string(nameof(typeof(stat)))
tb_name(o::Skip) = "Skip($(o.b))"
tb_name(o::Thin) = "Thin($(o.b))"
tb_name(o::WindowStat) = "WindowStat($(o.b))"
tb_name(o::AutoCov, b::Int) = "AutoCov(lag=$b)/corr"
tb_name(s1::String, s2::String) = s1 * "/" * s2
tb_name(arg1, arg2) = tb_name(arg1) * "/" * tb_name(arg2)
tb_name(arg, args...) = tb_name(arg) * "/" * tb_name(args...)

####################################
### TensorBoardLogger preprocess ###
####################################

function TBL.preprocess(name, stat::OnlineStat, data)
    return nobs(stat) > 0 && TBL.preprocess(tb_name(name, stat), value(stat), data)
end

function TBL.preprocess(name, stat::Skip, data)
    return TBL.preprocess(tb_name(name, stat), stat.stat, data)
end

function TBL.preprocess(name, stat::Thin, data)
    return TBL.preprocess(tb_name(name, stat), stat.stat, data)
end

function TBL.preprocess(name, stat::WindowStat, data)
    return TBL.preprocess(tb_name(name, stat), value(stat), data)
end

function TBL.preprocess(name, stat::AutoCov, data)
    autocor = OnlineStats.autocor(stat)
    for b in 1:(stat.lag.b - 1)
        bname = tb_name(stat, b)
        TBL.preprocess(tb_name(name, bname), autocor[b + 1], data)
    end
end

function TBL.preprocess(name, stat::Series, data)
    for s in stat.stats
        TBL.preprocess(name, s, data)
    end
end

function TBL.preprocess(name, hist::KHist, data)
    if nobs(hist) > 0
        edges = OnlineStats.edges(hist)
        cnts = OnlineStats.counts(hist)
        TBL.preprocess(name, (edges, cnts ./ sum(cnts)), data)
    end
end

function TBL.log_histogram(
    logger::AbstractLogger,
    name::AbstractString,
    hist::OnlineStats.HistogramStat;
    step=nothing,
    normalize=false,
)
    edges = OnlineStats.edges(hist)
    cnts = Float64.(OnlineStats.counts(hist))
    if normalize
        TBL.log_histogram(logger, name, (edges, cnts ./ sum(cnts)); step=step)
    else
        TBL.log_histogram(logger, name, (edges, cnts); step=step)
    end
end

###########################
### TensorBoardCallback ###
###########################

maybe_filter(f; kwargs...) = f
maybe_filter(::Nothing; exclude=nothing, include=nothing) = NameFilter(; exclude, include)

"""
    TensorBoardCallback

Wraps a `CoreLogging.AbstractLogger` to construct a callback to be
passed to `AbstractMCMC.step`.

# Usage

    TensorBoardCallback(; kwargs...)
    TensorBoardCallback(directory::String[, stats]; kwargs...)
    TensorBoardCallback(lg::AbstractLogger[, stats]; kwargs...)

Constructs an instance of a `TensorBoardCallback`, creating a `TBLogger` if `directory` is
provided instead of `lg`.

## Arguments
- `lg`: an instance of an `AbstractLogger` which implements `increment_step!`.
- `stats = nothing`: `OnlineStat` or lookup for variable name to statistic estimator.

## Keyword arguments
- `num_bins::Int = 100`: Number of bins to use in the histograms.
- `filter = nothing`: Filter determining whether or not we should log stats for a
  particular variable and value.
- `exclude = String[]`: If non-empty, these variables will not be logged.
- `include = String[]`: If non-empty, only these variables will be logged.
- `include_extras::Bool = true`: Include extra statistics from transitions.
- `include_hyperparams::Bool = true`: Include hyperparameters.
"""
struct TensorBoardCallback{L,P,F1,F2,F3}
    logger::AbstractLogger
    stats::L
    stat_prototype::P
    variable_filter::F1
    include_extras::Bool
    extras_filter::F2
    include_hyperparams::Bool
    hyperparam_filter::F3
    param_prefix::String
    extras_prefix::String
end

function TensorBoardCallback(directory::String, args...; kwargs...)
    return TensorBoardCallback(args...; directory=directory, kwargs...)
end

function TensorBoardCallback(args...; comment="", directory=nothing, kwargs...)
    log_dir = if isnothing(directory)
        "runs/$(format(now(), DateFormat("Y-m-d_H-M-S")))-$(gethostname())$(comment)"
    else
        directory
    end
    lg = TBLogger(log_dir; min_level=Info, step_increment=0)
    return TensorBoardCallback(lg, args...; kwargs...)
end

function TensorBoardCallback(
    lg::AbstractLogger,
    stats=nothing;
    num_bins::Int=100,
    exclude=nothing,
    include=nothing,
    filter=nothing,
    include_extras::Bool=true,
    extras_include=nothing,
    extras_exclude=nothing,
    extras_filter=nothing,
    include_hyperparams::Bool=false,
    hyperparams_include=nothing,
    hyperparams_exclude=nothing,
    hyperparams_filter=nothing,
    param_prefix::String="",
    extras_prefix::String="extras/",
    kwargs...,
)
    variable_filter_f = maybe_filter(filter; include=include, exclude=exclude)
    extras_filter_f = maybe_filter(
        extras_filter; include=extras_include, exclude=extras_exclude
    )
    hyperparams_filter_f = maybe_filter(
        hyperparams_filter; include=hyperparams_include, exclude=hyperparams_exclude
    )

    stats_dict, prototype = if stats isa OnlineStat
        OnlineStats.nobs(stats) > 0 &&
            @warn("using statistic with observations as a base: $(stats)")
        (Dict{String,typeof(stats)}(), deepcopy(stats))
    elseif !isnothing(stats)
        (stats, nothing)
    else
        o = OnlineStats.Series(Mean(), Variance(), KHist(num_bins))
        (Dict{String,typeof(o)}(), deepcopy(o))
    end

    return TensorBoardCallback(
        lg,
        stats_dict,
        prototype,
        variable_filter_f,
        include_extras,
        extras_filter_f,
        include_hyperparams,
        hyperparams_filter_f,
        param_prefix,
        extras_prefix,
    )
end

function filter_param_and_value(cb::TensorBoardCallback, param, value)
    return cb.variable_filter(param, value)
end
function filter_param_and_value(cb::TensorBoardCallback, param_and_value::Tuple)
    return filter_param_and_value(cb, param_and_value...)
end
function filter_param_and_value(cb::TensorBoardCallback, param_and_value::Pair)
    return filter_param_and_value(cb, first(param_and_value), last(param_and_value))
end

function filter_extras_and_value(cb::TensorBoardCallback, name, value)
    return cb.extras_filter(name, value)
end
function filter_extras_and_value(cb::TensorBoardCallback, name_and_value::Tuple)
    return filter_extras_and_value(cb, name_and_value...)
end
function filter_extras_and_value(cb::TensorBoardCallback, name_and_value::Pair)
    return filter_extras_and_value(cb, first(name_and_value), last(name_and_value))
end

function filter_hyperparams_and_value(cb::TensorBoardCallback, name, value)
    return cb.hyperparam_filter(name, value)
end
function filter_hyperparams_and_value(
    cb::TensorBoardCallback, name_and_value::Union{Pair,Tuple}
)
    return filter_hyperparams_and_value(cb, name_and_value...)
end

increment_step!(lg::TBLogger, Δ_Step) = TensorBoardLogger.increment_step!(lg, Δ_Step)

function (cb::TensorBoardCallback)(
    rng, model, sampler, transition, state, iteration; kwargs...
)
    stats = cb.stats
    lg = cb.logger
    variable_filter = Base.Fix1(filter_param_and_value, cb)
    extras_filter = Base.Fix1(filter_extras_and_value, cb)
    hyperparams_filter = Base.Fix1(filter_hyperparams_and_value, cb)

    if iteration == 1 && cb.include_hyperparams
        hparams = Dict(
            Iterators.filter(
                hyperparams_filter,
                AbstractMCMC.hyperparams(model, sampler, state; kwargs...),
            ),
        )
        if !isempty(hparams)
            TensorBoardLogger.write_hparams!(
                lg, hparams, AbstractMCMC.hyperparam_metrics(model, sampler)
            )
        end
    end

    with_logger(lg) do
        for (k, val) in Iterators.filter(
            variable_filter,
            AbstractMCMC.params_and_values(model, sampler, transition, state; kwargs...),
        )
            stat = if stats isa AbstractDict && cb.stat_prototype !== nothing
                get!(stats, k) do
                    deepcopy(cb.stat_prototype)
                end
            elseif stats isa AbstractDict
                get(stats, k, nothing)
            else
                nothing
            end

            @info "$(cb.param_prefix)$k" val
            if stat !== nothing
                OnlineStats.fit!(stat, val)
                @info "$(cb.param_prefix)$k" stat
            end
        end

        if cb.include_extras
            for (name, val) in Iterators.filter(
                extras_filter,
                AbstractMCMC.extras(model, sampler, transition, state; kwargs...),
            )
                @info "$(cb.extras_prefix)$(name)" val
                if val isa Real
                    # Extras often don't have a pre-defined list, so we might need to create stats on the fly.
                    # We reuse the prototype if available.
                    stat = if stats isa AbstractDict && cb.stat_prototype !== nothing
                        get!(stats, "$(cb.extras_prefix)$(name)") do
                            deepcopy(cb.stat_prototype)
                        end
                    elseif stats isa AbstractDict
                        get(stats, "$(cb.extras_prefix)$(name)", nothing)
                    else
                        nothing
                    end

                    if stat !== nothing
                        fit!(stat, float(val))
                        @info ("$(cb.extras_prefix)$(name)") stat
                    end
                end
            end
        end
        increment_step!(lg, 1)
    end
end

end
