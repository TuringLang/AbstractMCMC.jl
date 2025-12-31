"""
    SamplingStats

A struct that tracks sampling information. 

The fields available are:

- `start`: A `Float64` Unix timestamp indicating the start time of sampling.
- `stop`: A `Float64` Unix timestamp indicating the stop time of sampling.
- `duration`: The sampling time duration, defined as `stop - start`.
"""
struct SamplingStats
    start::Float64
    stop::Float64
    duration::Float64
end

"""
    Skip(b::Int, stat)

Wrapper that skips the first `b` observations before passing them on to `stat`.
Does not require `OnlineStats` to be loaded, but implements the interface if available.
"""
mutable struct Skip{S}
    b::Int
    n::Int
    stat::S
end
Skip(b::Int, stat) = Skip(b, 0, stat)

"""
    Thin(b::Int, stat)

Wrapper that thins the stream of observations by a factor of `b`.
Only every `b`-th observation is passed to `stat`.
Does not require `OnlineStats` to be loaded, but implements the interface if available.
"""
mutable struct Thin{S}
    b::Int
    n::Int
    stat::S
end
Thin(b::Int, stat) = Thin(b, 0, stat)

"""
    WindowStat(b::Int, stat; T=Float64)

Wrapper that calculates `stat` over a sliding window of size `b`.
Internal buffer stores elements of type `T` (defaults to `Float64`).
Does not require `OnlineStats` to be loaded, but implements the interface if available.

Note: This reimplements the logic of `OnlineStats.MovingWindow` to avoid a hard dependency.
"""
mutable struct WindowStat{T,S}
    b::Int
    n::Int
    stat::S
    buffer::Vector{T}
end

function WindowStat(b::Int, stat::S; T::Type=Float64) where {S}
    return WindowStat{T,S}(b, 0, stat, Vector{T}(undef, b))
end
