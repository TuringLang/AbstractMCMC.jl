module AbstractMCMCOnlineStatsExt

using AbstractMCMC
using OnlineStats
using OnlineStats: OnlineStats, OnlineStat, fit!, value, nobs, MovingWindow

##########################
### Skip <: OnlineStat ###
##########################

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

Base.show(io::IO, o::Skip) = print(io, "Skip ($(o.b)): n=$(o.n) | stat=$(o.stat)")

##########################
### Thin <: OnlineStat ###
##########################

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

Base.show(io::IO, o::Thin) = print(io, "Thin ($(o.b)): n=$(o.n) | stat=$(o.stat)")

################################
### WindowStat <: OnlineStat ###
################################

"""
    WindowStat(b::Int, stat::OnlineStat)

"Wraps" `stat` in a `MovingWindow` of length `b`.
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
    print(io, "WindowStat ($(o.window.b)): nobs=$(nobs(o)) | stat=$(o.stat)")
end

end
