module AbstractMCMCTensorBoardLoggerExt

using AbstractMCMC
using AbstractMCMC:
    MultiCallback,
    NameFilter,
    _names_and_values,
    hyperparam_metrics,
    merge_with_defaults,
    create_stats_with_options,
    DEFAULT_STATS_OPTIONS,
    DEFAULT_NAME_FILTER
using TensorBoardLogger
using TensorBoardLogger: TBLogger
using Logging: AbstractLogger, with_logger, @info

"""
    TensorBoardCallback

A callback for logging MCMC samples to TensorBoard.
Supports statistics collection when OnlineStats.jl is loaded.
"""
struct TensorBoardCallback{L,S,P,F}
    logger::L
    stats::S
    stat_prototype::P
    variable_filter::F
    include_extras::Bool
    include_hyperparams::Bool
    param_prefix::String
    extras_prefix::String
end

"""
    mcmc_callback(; logger, stats=nothing, stats_options=nothing, name_filter=nothing, num_bins=100)

Create a TensorBoard logging callback.

# Arguments
- `logger`: An `AbstractLogger` instance (e.g., `TBLogger` from TensorBoardLogger.jl)
- `stats`: Statistics to collect. Can be:
  - `nothing`: No statistics (default)
  - `true` or `:default`: Use default statistics (Mean, Variance, KHist) - requires OnlineStats
  - An OnlineStat or tuple of OnlineStats - requires OnlineStats
- `stats_options`: NamedTuple with `thin`, `skip`, `window`
- `name_filter`: NamedTuple with `include`, `exclude`, `extras`, `hyperparams`
- `num_bins`: Number of histogram bins (default: 100)

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
"""
function AbstractMCMC.mcmc_callback(;
    logger::AbstractLogger,
    stats=nothing,
    stats_options=nothing,
    name_filter=nothing,
    num_bins::Int=100,
)
    merged_stats_options = merge_with_defaults(stats_options, DEFAULT_STATS_OPTIONS)
    merged_name_filter = merge_with_defaults(name_filter, DEFAULT_NAME_FILTER)

    processed_stats = create_stats_with_options(stats, merged_stats_options, num_bins)

    variable_filter = NameFilter(;
        include=merged_name_filter.include, exclude=merged_name_filter.exclude
    )

    stats_dict, prototype = if processed_stats === nothing
        (nothing, nothing)
    else
        processed_stats
    end

    callback = TensorBoardCallback(
        logger,
        stats_dict,
        prototype,
        variable_filter,
        merged_name_filter.extras,
        merged_name_filter.hyperparams,
        "",
        "extras/",
    )

    return MultiCallback((callback,))
end

function filter_name_and_value(cb::TensorBoardCallback, name_and_value)
    return cb.variable_filter(first(name_and_value), last(name_and_value))
end

function (cb::TensorBoardCallback)(
    rng, model, sampler, transition, state, iteration; kwargs...
)
    stats = cb.stats
    lg = cb.logger
    filter_fn = Base.Fix1(filter_name_and_value, cb)

    if iteration == 1 && cb.include_hyperparams
        hp_iter = _names_and_values(
            model,
            sampler,
            transition,
            state;
            params=false,
            hyperparams=true,
            extra=false,
            kwargs...,
        )
        hparams = Dict(hp_iter)
        if !isempty(hparams)
            TensorBoardLogger.write_hparams!(
                lg, hparams, AbstractMCMC.hyperparam_metrics(model, sampler)
            )
        end
    end

    with_logger(lg) do
        all_values = _names_and_values(
            model,
            sampler,
            transition,
            state;
            params=true,
            hyperparams=false,
            extra=cb.include_extras,
            kwargs...,
        )

        for (k, val) in Iterators.filter(filter_fn, all_values)
            @info "$(cb.param_prefix)$k" val

            if stats !== nothing
                _log_stat!(stats, cb.stat_prototype, k, val, cb.param_prefix)
            end
        end

        TensorBoardLogger.increment_step!(lg, 1)
    end
end

function _log_stat!(stats, prototype, key, val, prefix)
    ext = Base.get_extension(AbstractMCMC, :AbstractMCMCOnlineStatsExt)
    if ext !== nothing
        ext.log_stat_impl!(stats, prototype, key, val, prefix)
    end
end

end
