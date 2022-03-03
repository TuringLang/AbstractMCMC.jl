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
