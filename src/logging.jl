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
    if (Sys.iswindows() && VERSION < v"1.5.3") || (isdefined(Main, :IJulia) && Main.IJulia.inited)
        return ConsoleProgressMonitor.ProgressLogger()
    else
        return TerminalLoggers.TerminalLogger()
    end
end

##

using ProgressLogging: _asprogress, ProgressString, ProgressLevel, uuid4
using Base.Meta: isexpr

const _id_var_children = gensym(:progress_id_children)
const _name_var_children = gensym(:progress_name_children)

macro progressid_child(N)
    id_err = "`@progressid_child` must be used inside `@withprogress_children`"

    quote
        $Base.@isdefined($_id_var_children) ? $_id_var_children[$N] : $error($id_err)
    end |> esc
end

macro logprogress_child(args...)
    _logprogress_child(args...)
end

function _logprogress_child(N, name, progress = nothing, args...)
    name_expr = :($Base.@isdefined($_name_var_children) ? $_name_var_children[$N] : "")
    if progress == nothing
        # Handle: @logprogress progress
        kwargs = (:(progress = $name), args...)
        progress = name
        name = name_expr
    elseif isexpr(progress, :(=)) && progress.args[1] isa Symbol
        # Handle: @logprogress progress key1=val1 ...
        kwargs = (:(progress = $name), progress, args...)
        progress = name
        name = name_expr
    else
        # Otherwise, it's: @logprogress name progress key1=val1 ...
        kwargs = (:(progress = $progress), args...)
    end

    id_err = "`@logprogress_child` must be used inside `@withprogress_children`"
    id_expr = :($Base.@isdefined($_id_var_children) ? $_id_var_children[$N] : $error($id_err))

    @gensym id_tmp
    # Emitting progress log record as old/open API (i.e., using
    # `progress` key) and _also_ as new API based on `Progress` type.
    msgexpr = :($ProgressString($_asprogress(
        $name,
        $id_tmp,
        $(ProgressLogging._id_var);
        progress = $progress,
    )))
    quote
        $id_tmp = $id_expr
        $Logging.@logmsg($ProgressLevel, $msgexpr, $(kwargs...), _id = $id_tmp)
    end |> esc
end

macro withprogress_children(N, exprs...)
    _withprogress_children(N, exprs...)
end

function _withprogress_children(N, exprs...)
    length(exprs) == 0 &&
        throw(ArgumentError("`@withprogress_children` requires at least one number and one expression."))

    m = ProgressLogging.@__MODULE__

    kwargs = Dict{Symbol,Any}(:names => :(["" for i in 1:$N]))
    unsupported = []
    for kw in exprs[1:end-1]
        if isexpr(kw, :(=)) && length(kw.args) == 2 && haskey(kwargs, kw.args[1])
            kwargs[kw.args[1]] = kw.args[2]
        else
            push!(unsupported, kw)
        end
    end

    # Error on invalid input expressions:
    if !isempty(unsupported)
        msg = sprint() do io
            println(io, "Unsupported optional arguments:")
            for kw in unsupported
                println(io, kw)
            end
            print(io, "`@withprogress_children` supports only following keyword arguments: ")
            join(io, keys(kwargs), ", ")
        end
        throw(ArgumentError(msg))
    end

    ex = exprs[end]
    id_err = "`@withprogress_children` must be used inside `@withprogress`"

    i_var = gensym()
    quote
        let 
            $Base.@isdefined($(ProgressLogging._id_var)) || $error($id_err)
            $_id_var_children = [ $uuid4() for i in 1:$N ]
            $_name_var_children = $(kwargs[:names])

            for $i_var in 1:$N
                @logprogress_child $i_var nothing
            end

            try
                $ex
            finally
                for $i_var in 1:$N
                    @logprogress_child $i_var nothing
                end
            end
        end
    end |> esc
end

macro ifwithprogresslogger_children(progress, exprs...)
    return quote
        if $progress
            if $hasprogresslevel($Logging.current_logger())
                @withprogress_children $(exprs...)
            else
                $with_progresslogger($Base.@__MODULE__, $Logging.current_logger()) do
                    @withprogress_children $(exprs...)
                end
            end
        else
            $(exprs[end])
        end
    end |> esc
end


