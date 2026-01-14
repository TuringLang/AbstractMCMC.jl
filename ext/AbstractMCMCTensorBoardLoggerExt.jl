module AbstractMCMCTensorBoardLoggerExt

using AbstractMCMC
using AbstractMCMC: NameFilter, _names_and_values, hyperparam_metrics
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
    create_tensorboard_callback(logger; stats, name_filter)

Create a TensorBoardCallback. Called from `AbstractMCMC.mcmc_callback`.
"""
function create_tensorboard_callback(logger::AbstractLogger; stats, name_filter)
    variable_filter = NameFilter(;
        include=isempty(name_filter.include) ? nothing : name_filter.include,
        exclude=isempty(name_filter.exclude) ? nothing : name_filter.exclude,
    )

    stats_dict, prototype = if stats === nothing
        (nothing, nothing)
    else
        stats
    end

    return TensorBoardCallback(
        logger,
        stats_dict,
        prototype,
        variable_filter,
        name_filter.extras,
        name_filter.hyperparams,
        "",
        "extras/",
    )
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
