"""
    SamplingStats

A struct that tracks sampling information. 

The fields available are:

- `start`: A `Float64` Unix timestamp indicating the start time of sampling.
- `stop`: A `Float64` Unix timestamp indicating the stop time of sampling.
- `duration`: The sampling time duration, defined as `stop - start`.
- `allocations`: The total number of bytes allocated by the sampling process.
"""
struct SamplingStats
    start::Float64
    stop::Float64
    duration::Float64
    allocations::Int64
end