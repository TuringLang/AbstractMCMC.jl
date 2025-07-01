"""
    AbstractProgressKwarg

Abstract type representing the values that the `progress` keyword argument can
internally take for single-chain sampling.
"""
abstract type AbstractProgressKwarg end

"""
    CreateNewProgressBar

Create a new logger for progress logging.
"""
struct CreateNewProgressBar{S<:AbstractString} <: AbstractProgressKwarg
    name::S
    uuid::UUIDs.UUID

    function CreateNewProgressBar(name::AbstractString)
        return new{typeof{name}}(name, UUIDs.uuid4())
    end
end
function init_progress(p::CreateNewProgressBar)
    if hasprogresslevel(Logging.current_logger())
        ProgressLogging.@withprogress $(exprs...)
    else
        $with_progresslogger($Base.@__MODULE__, $Logging.current_logger()) do
            $ProgressLogging.@withprogress $(exprs...)
        end
    end
    ProgressLogging.@logprogress p.name nothing _id = p.uuid
end
function update_progress(p::CreateNewProgressBar, progress_frac, ::Bool)
    ProgressLogging.@logprogress p.name progress_frac _id = p.uuid
end
finish_progress(::CreateNewProgressBar) = ProgressLogging.@logprogress "done"

"""
    NoLogging

Do not log progress at all.
"""
struct NoLogging <: AbstractProgressKwarg end
init_progress(::NoLogging) = nothing
update_progress(::NoLogging, ::Any, ::Bool) = nothing
finish_progress(::NoLogging) = nothing

"""
    ExistingProgressBar

Use an existing progress bar to log progress. This is used for tracking
progress in a progress bar that has been previously generated elsewhere,
specifically, when `sample(..., MCMCThreads(), ...; progress=:perchain)` is
called. In this case we can use `@logprogress name progress_frac _id = uuid` to
log progress.
"""
struct ExistingProgressBar{S<:AbstractString} <: AbstractProgressKwarg
    name::S
    uuid::UUIDs.UUID
end
init_progress(::ExistingProgressBar) = nothing
function update_progress(p::ExistingProgressBar, progress_frac, ::Bool)
    ProgressLogging.@logprogress p.name progress_frac _id = p.uuid
end
function finish_progress(p::ExistingProgressBar)
    ProgressLogging.@logprogress p.name "done" _id = p.uuid
end

"""
    ChannelProgress

Use a `Channel` to log progress. This is used for 'reporting' progress back
to the main thread or worker when using `progress=:overall` with MCMCThreads or
MCMCDistributed.
"""
struct ChannelProgress{T<:Union{Channel{Bool},Distributed.RemoteChannel{Channel{Bool}}}} <:
       AbstractProgressKwarg
    channel::T
end
init_progress(::ChannelProgress) = nothing
function update_progress(p::ChannelProgress, ::Any, update_channel::Bool)
    return update_channel && put!(p.channel, true)
end
# Note: We don't want to `put!(p.channel, false)`, because that would stop the
# channel from being used for further updates e.g. from other chains.
finish_progress(::ChannelProgress) = nothing

# avoid creating a progress bar with @withprogress if progress logging is disabled
# and add a custom progress logger if the current logger does not seem to be able to handle
# progress logs
macro ifwithprogresslogger(cond, exprs...)
    return esc(
        quote
            if $cond
                # Create a new logger
                if $hasprogresslevel($Logging.current_logger())
                    $ProgressLogging.@withprogress $(exprs...)
                else
                    $with_progresslogger($Base.@__MODULE__, $Logging.current_logger()) do
                        $ProgressLogging.@withprogress $(exprs...)
                    end
                end
            else
                # Don't create a new logger, either because progress logging
                # was disabled, or because it's otherwise being manually
                # managed.
                $(exprs[end])
            end
        end,
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
