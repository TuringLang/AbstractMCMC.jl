using Dates

# Tracks sampling metadata
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