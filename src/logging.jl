"""
    AbstractProgressKwarg

Abstract type representing the values that the `progress` keyword argument can
internally take for single-chain sampling.
"""
abstract type AbstractProgressKwarg end

DEFAULT_N_UPDATES = 200

"""
    CreateNewProgressBar

Create a new logger for progress logging.
"""
struct CreateNewProgressBar{S<:AbstractString} <: AbstractProgressKwarg
    name::S
    uuid::UUIDs.UUID
    function CreateNewProgressBar(name::AbstractString)
        return new{typeof(name)}(name, UUIDs.uuid4())
    end
end
function init_progress!(p::CreateNewProgressBar)
    ProgressLogging.@logprogress p.name nothing _id = p.uuid
end
function update_progress!(p::CreateNewProgressBar, progress_frac)
    ProgressLogging.@logprogress p.name progress_frac _id = p.uuid
end
function finish_progress!(p::CreateNewProgressBar)
    ProgressLogging.@logprogress p.name "done" _id = p.uuid
end
get_n_updates(::CreateNewProgressBar) = DEFAULT_N_UPDATES

"""
    NoLogging

Do not log progress at all.
"""
struct NoLogging <: AbstractProgressKwarg end
init_progress!(::NoLogging) = nothing
update_progress!(::NoLogging, ::Any) = nothing
finish_progress!(::NoLogging) = nothing
get_n_updates(::NoLogging) = DEFAULT_N_UPDATES

"""
    ExistingProgressBar
Use an existing progress bar to log progress. This is used for tracking
progress in a progress bar that has been previously generated elsewhere,
specifically, during multi-threaded sampling where per-chain progress
bars are requested. In this case we can use `@logprogress name progress_frac
_id = uuid` to log progress.
"""
struct ExistingProgressBar{S<:AbstractString} <: AbstractProgressKwarg
    name::S
    uuid::UUIDs.UUID
end
function init_progress!(p::ExistingProgressBar)
    # Hacky code to reset the start timer if called from a multi-chain sampling
    # process. We need this because the progress bar is constructed in the
    # multi-chain method, i.e. if we don't do this the progress bar shows the
    # time elapsed since _all_ sampling began, not since the current chain
    # started.
    try
        bartrees = Logging.current_logger().loggers[1].logger.bartrees
        bar = TerminalLoggers.findbar(bartrees, p.uuid).data
        bar.tfirst = time()
    catch
    end
    ProgressLogging.@logprogress p.name nothing _id = p.uuid
end
function update_progress!(p::ExistingProgressBar, progress_frac)
    ProgressLogging.@logprogress p.name progress_frac _id = p.uuid
end
function finish_progress!(p::ExistingProgressBar)
    ProgressLogging.@logprogress p.name "done" _id = p.uuid
end
get_n_updates(::ExistingProgressBar) = DEFAULT_N_UPDATES

"""
    ChannelProgress

Use a `Channel` to log progress. This is used for 'reporting' progress back to
the main thread or worker when using multi-threaded or distributed sampling.

n_updates is the number of updates that each child thread is expected to report
back to the main thread.
"""
struct ChannelProgress{T<:Union{Channel{Bool},Distributed.RemoteChannel{Channel{Bool}}}} <:
       AbstractProgressKwarg
    channel::T
    n_updates::Int
end
init_progress!(::ChannelProgress) = nothing
update_progress!(p::ChannelProgress, ::Any) = put!(p.channel, true)
# Note: We don't want to `put!(p.channel, false)`, because that would stop the
# channel from being used for further updates e.g. from other chains.
finish_progress!(::ChannelProgress) = nothing
get_n_updates(p::ChannelProgress) = p.n_updates

"""
    ChannelPlusExistingProgress

Send updates to two places: a `Channel` as well as an existing progress bar.
"""
struct ChannelPlusExistingProgress{C<:ChannelProgress,E<:ExistingProgressBar} <:
       AbstractProgressKwarg
    channel_progress::C
    existing_progress::E
end
function init_progress!(p::ChannelPlusExistingProgress)
    init_progress!(p.channel_progress)
    init_progress!(p.existing_progress)
    return nothing
end
function update_progress!(p::ChannelPlusExistingProgress, progress_frac)
    update_progress!(p.channel_progress, progress_frac)
    update_progress!(p.existing_progress, progress_frac)
    return nothing
end
function finish_progress!(p::ChannelPlusExistingProgress)
    finish_progress!(p.channel_progress)
    finish_progress!(p.existing_progress)
    return nothing
end
get_n_updates(p::ChannelPlusExistingProgress) = get_n_updates(p.channel_progress)

# Add a custom progress logger if the current logger does not seem to be able to handle
# progress logs.
macro maybewithricherlogger(expr)
    return esc(
        quote
            if !($hasprogresslevel($Logging.current_logger()))
                $with_progresslogger($Base.@__MODULE__, $Logging.current_logger()) do
                    $(expr)
                end
            else
                $(expr)
            end
        end
    )
end

# improved checks?
function hasprogresslevel(logger)
    return Logging.min_enabled_level(logger) ≤ ProgressLogging.ProgressLevel
end

# filter better, e.g., according to group?
function with_progresslogger(f, _module, logger)
    logger1 = LoggingExtras.EarlyFilteredLogger(progresslogger()) do log
        log._module === _module && log.level == ProgressLogging.ProgressLevel
    end
    logger2 = LoggingExtras.EarlyFilteredLogger(logger) do log
        log._module !== _module || log.level != ProgressLogging.ProgressLevel
    end

    return Logging.with_logger(f, LoggingExtras.TeeLogger(logger1, logger2))
end

function progresslogger()
    # detect if code is running under IJulia since TerminalLogger does not work with IJulia
    # https://github.com/JuliaLang/IJulia.jl#detecting-that-code-is-running-under-ijulia
    if (isdefined(Main, :IJulia) && Main.IJulia.inited)
        return ConsoleProgressMonitor.ProgressLogger()
    else
        return TerminalLoggers.TerminalLogger()
    end
end
