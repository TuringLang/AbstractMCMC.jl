using Dates

"""
    Metadata

A struct that tracks sampling information. 

The fields available are:

- `start_time`: A `DateTime` indicating when sampling begun.
- `stop_time`: A `DateTime` indicating when sampling finished.
- `step_calls`: The number of times `step!` was called.
- `step_time`: The total number of seconds spent inside `step!` calls.
- `allocations`: The total number of bytes allocated by `step!`.
"""
mutable struct Metadata
    start_time::DateTime
    stop_time::Union{DateTime, Missing}
    step_time::Float64
    step_calls::Int64
    allocations::Int64
end

Metadata() = Metadata(Dates.now(), missing, 0,0,0)

function update(f, md::Metadata)
    value, stats... = @timed f(md)

    md.step_time += stats.time
    md.step_calls += 1
    md.allocations += stats.bytes

    return value
end

function stop(md::Metadata)
    md.stop_time = Dates.now()
end