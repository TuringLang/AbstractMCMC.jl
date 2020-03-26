# avoid creating a progress bar with @withprogress if progress logging is disabled
# and add a custom progress logger if the current logger does not seem to be able to handle
# progress logs
macro ifwithprogresslogger(progress, exprs...)
    return quote
        if $progress
            if $hasprogresslevel($Logging.current_logger())
                $ProgressLogging.@withprogress $(exprs...)
            else
                $with_progresslogger($Base.@__MODULE__, $Logging.current_logger()) do
                    $ProgressLogging.@withprogress $(exprs...)
                end
            end
        else
            $(exprs[end])
        end
    end |> esc
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

    Logging.with_logger(f, LoggingExtras.TeeLogger(logger1, logger2))
end

function progresslogger()
    # detect if code is running under IJulia since TerminalLogger does not work with IJulia
    # https://github.com/JuliaLang/IJulia.jl#detecting-that-code-is-running-under-ijulia
    if Sys.iswindows() || (isdefined(Main, :IJulia) && Main.IJulia.inited)
        return ConsoleProgressMonitor.ProgressLogger()
    else
        return TerminalLoggers.TerminalLogger()
    end
end
