module AbstractMCMCTensorBoardLoggerExt

using AbstractMCMC
using AbstractMCMC: NameFilter, params_and_values, extras, hyperparams, hyperparam_metrics
using TensorBoardLogger
using TensorBoardLogger: TBLogger
using OnlineStats
using OnlineStats:
    OnlineStats,
    OnlineStat,
    Mean,
    Variance,
    KHist,
    Series,
    AutoCov,
    MovingWindow,
    fit!,
    value,
    nobs
using Logging: AbstractLogger, with_logger, @info, Info
using Dates: now, DateFormat, format

const TBL = TensorBoardLogger

# Import Skip/Thin/WindowStat from OnlineStatsExt (loaded when OnlineStats loaded)
# Since this extension requires both TBL AND OnlineStats, OnlineStatsExt is already loaded
const OnlineStatsExt = Base.get_extension(AbstractMCMC, :AbstractMCMCOnlineStatsExt)
const Skip = OnlineStatsExt.Skip
const Thin = OnlineStatsExt.Thin
const WindowStat = OnlineStatsExt.WindowStat

#########################################
### TensorBoardLogger name formatting ###
#########################################

tb_name(arg) = string(arg)
tb_name(stat::OnlineStat) = string(nameof(typeof(stat)))
tb_name(o::Skip) = "Skip($(o.b))"
tb_name(o::Thin) = "Thin($(o.b))"
tb_name(o::WindowStat) = "WindowStat($(o.window.b))"
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
        # `autocor[i]` corresponds to the lag of size `i - 1` and `autocor[1] = 1.0`
        bname = tb_name(stat, b)
        TBL.preprocess(tb_name(name, bname), autocor[b + 1], data)
    end
end

function TBL.preprocess(name, stat::Series, data)
    # Iterate through the stats and process those independently
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

"""
    create_tensorboard_callback(logdir; logger, stats, stats_options, name_filter)

Create a TensorBoardCallback from the new unified API arguments.
Called from AbstractMCMC.mcmc_callback.

If `logger` is provided, uses it directly. Otherwise creates a TBLogger from `logdir`.
"""
function create_tensorboard_callback(
    logdir; logger=nothing, stats, stats_options, name_filter, num_bins::Int=100
)
    # Use provided logger or create TBLogger from logdir
    lg = if logger !== nothing
        logger
    else
        log_dir = if isnothing(logdir)
            "runs/$(format(now(), DateFormat("Y-m-d_H-M-S")))-$(gethostname())"
        else
            logdir
        end
        TBLogger(log_dir; min_level=Info, step_increment=0)
    end

    # Create variable filter from name_filter
    variable_filter = NameFilter(;
        include=isempty(name_filter.include) ? nothing : name_filter.include,
        exclude=isempty(name_filter.exclude) ? nothing : name_filter.exclude,
    )

    # Extras and hyperparams filters
    extras_filter = NameFilter()
    hyperparams_filter = NameFilter()

    # Process stats with stats_options wrappers
    processed_stats = create_stats_with_options(stats, stats_options, num_bins)
    stats_dict, prototype = processed_stats

    return TensorBoardCallback(
        lg,
        stats_dict,
        prototype,
        variable_filter,
        name_filter.extras,
        extras_filter,
        name_filter.hyperparams,
        hyperparams_filter,
        "",  # param_prefix
        "extras/",  # extras_prefix
    )
end

"""
    create_stats_with_options(stats, stats_options, num_bins)

Create stats dictionary and prototype, applying Skip/Thin/WindowStat wrappers.
"""
function create_stats_with_options(stats, stats_options, num_bins)
    # Get base stat prototype
    base_stat = if stats isa OnlineStat
        stats
    elseif stats isa Tuple
        Series(stats...)
    elseif stats === nothing
        Series(Mean(), Variance(), KHist(num_bins))
    else
        stats
    end

    # Apply wrappers based on stats_options
    wrapped_stat = wrap_stat(base_stat, stats_options)

    if wrapped_stat isa OnlineStat
        nobs(wrapped_stat) > 0 &&
            @warn("using statistic with observations as a base: $(wrapped_stat)")
        return (Dict{String,typeof(wrapped_stat)}(), deepcopy(wrapped_stat))
    else
        return (wrapped_stat, nothing)
    end
end

"""
    wrap_stat(stat, stats_options)

Apply Skip, Thin, and WindowStat wrappers to a statistic based on options.
"""
function wrap_stat(stat, stats_options)
    result = stat

    # Apply window first (innermost)
    if stats_options.window < typemax(Int)
        result = WindowStat(stats_options.window, result)
    end

    # Apply skip
    if stats_options.skip > 0
        result = Skip(stats_options.skip, result)
    end

    # Apply thin (outermost)
    if stats_options.thin > 0
        result = Thin(stats_options.thin, result)
    end

    return result
end

# Legacy constructors
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

maybe_filter(f; kwargs...) = f
maybe_filter(::Nothing; exclude=nothing, include=nothing) = NameFilter(; exclude, include)

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

###############################
### Callback implementation ###
###############################

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
