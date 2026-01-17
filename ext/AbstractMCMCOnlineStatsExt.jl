module AbstractMCMCOnlineStatsExt

using AbstractMCMC
using OnlineStats
using OnlineStats:
    OnlineStat, fit!, value, nobs, MovingWindow, Series, Mean, Variance, KHist
using Logging: @info

"""
    Skip(b::Int, stat::OnlineStat)

Skips the first `b` observations before passing them on to `stat`.
"""
mutable struct Skip{T,O<:OnlineStat{T}} <: OnlineStat{T}
    b::Int
    n::Int
    stat::O
end

Skip(b::Int, stat::OnlineStat{T}) where {T} = Skip{T,typeof(stat)}(b, 0, stat)

OnlineStats.nobs(o::Skip) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Skip) = OnlineStats.value(o.stat)

function OnlineStats._fit!(o::Skip, x)
    if o.n >= o.b
        OnlineStats._fit!(o.stat, x)
    end
    o.n += 1
    return o
end

Base.show(io::IO, o::Skip) = print(io, "Skip($(o.b)): n=$(o.n) | $(o.stat)")

"""
    Thin(b::Int, stat::OnlineStat)

Thins `stat` with an interval `b`, i.e. only passes every b-th observation to `stat`.
"""
mutable struct Thin{T,O<:OnlineStat{T}} <: OnlineStat{T}
    b::Int
    n::Int
    stat::O
end

Thin(b::Int, stat::OnlineStat{T}) where {T} = Thin{T,typeof(stat)}(b, 0, stat)

OnlineStats.nobs(o::Thin) = OnlineStats.nobs(o.stat)
OnlineStats.value(o::Thin) = OnlineStats.value(o.stat)

function OnlineStats._fit!(o::Thin, x)
    if (o.n % o.b) == 0
        OnlineStats._fit!(o.stat, x)
    end
    o.n += 1
    return o
end

Base.show(io::IO, o::Thin) = print(io, "Thin($(o.b)): n=$(o.n) | $(o.stat)")

"""
    WindowStat(b::Int, stat::OnlineStat)

Wraps `stat` in a `MovingWindow` of length `b`.
"""
struct WindowStat{T,O} <: OnlineStat{T}
    window::MovingWindow{T}
    stat::O
end

function WindowStat(b::Int, stat::OnlineStat{T}) where {T}
    return WindowStat{T,typeof(stat)}(MovingWindow(b, T), stat)
end

OnlineStats.nobs(o::WindowStat) = OnlineStats.nobs(o.window)

function OnlineStats._fit!(o::WindowStat, x)
    OnlineStats._fit!(o.window, x)
    return o
end

function OnlineStats.value(o::WindowStat)
    stat_new = deepcopy(o.stat)
    fit!(stat_new, OnlineStats.value(o.window))
    return stat_new
end

function Base.show(io::IO, o::WindowStat)
    return print(io, "WindowStat($(o.window.b)): nobs=$(nobs(o)) | $(o.stat)")
end

"""
    create_stats_with_options_impl(stats, stats_options, num_bins)

Create stats dictionary and prototype, applying Skip/Thin/WindowStat wrappers.
"""
function create_stats_with_options_impl(stats, stats_options, num_bins)
    base_stat = if stats === true || stats === :default
        Series(Mean(), Variance(), KHist(num_bins))
    elseif stats isa OnlineStat
        stats
    elseif stats isa Tuple
        Series(stats...)
    else
        Series(Mean(), Variance(), KHist(num_bins))
    end

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

    if stats_options.window < typemax(Int)
        result = WindowStat(stats_options.window, result)
    end

    if stats_options.skip > 0
        result = Skip(stats_options.skip, result)
    end

    if stats_options.thin > 0
        result = Thin(stats_options.thin, result)
    end

    return result
end

"""
    log_stat_impl!(stats, prototype, key, val, prefix)

Update and log statistics. Called from TensorBoard callback.
"""
function log_stat_impl!(stats::AbstractDict, prototype, key, val, prefix)
    # Safety: extract value if val is a Pair (can happen with nested iteration)
    actual_val = val isa Pair ? last(val) : val

    if !(actual_val isa Real)
        return nothing
    end
    float_val = Float64(actual_val)

    str_key = string(key)

    stat = if prototype !== nothing
        get!(stats, str_key) do
            deepcopy(prototype)
        end
    else
        get(stats, str_key, nothing)
    end

    if stat !== nothing
        fit!(stat, float_val)
        @info "$(prefix)$str_key" stat
    end
end

log_stat_impl!(::Nothing, prototype, key, val, prefix) = nothing

# tb_name helpers for formatting stat names in TensorBoard
tb_name(arg) = string(arg)
tb_name(stat::OnlineStat) = string(nameof(typeof(stat)))
tb_name(o::Skip) = "Skip($(o.b))"
tb_name(o::Thin) = "Thin($(o.b))"
tb_name(o::WindowStat) = "WindowStat($(o.window.b))"
tb_name(s1::String, s2::String) = s1 * "/" * s2
tb_name(arg1, arg2) = tb_name(arg1) * "/" * tb_name(arg2)
tb_name(arg, args...) = tb_name(arg) * "/" * tb_name(args...)

end
