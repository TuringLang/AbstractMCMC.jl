"""
    SamplingStats

A struct that tracks sampling information. 

The fields available are:

- `start_time`: A `DateTime` indicating when sampling begun.
- `stop_time`: A `DateTime` indicating when sampling finished.
- `step_calls`: The number of times `step!` was called.
- `step_time`: The total number of seconds spent inside `step!` calls.
- `allocations`: The total number of bytes allocated by `step!`.
"""
struct SamplingStats
    start::Float64
    stop::Union{Float64, Missing}
    duration::Union{Float64, Missing}
    step_time::Float64
    step_calls::Int64
    allocations::Int64
end

function update(f, md::SamplingStats)
    (sample, state), etime, alloc, gct, _ = @timed f(md)

    md.step_time += etime
    md.step_calls += 1
    md.allocations += alloc

    return (sample, state)
end

function stop(md::SamplingStats)
    md.stop_time = Dates.now()
end