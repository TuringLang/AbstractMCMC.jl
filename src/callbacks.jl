"""
    MultiCallback

A callback that combines multiple callbacks into one.

Implements `push!!` from [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl) to add callbacks to the list.
"""
struct MultiCallback{Cs}
    callbacks::Cs
end

MultiCallback() = MultiCallback(())
MultiCallback(callbacks...) = MultiCallback(callbacks)

(c::MultiCallback)(args...; kwargs...) = foreach(c -> c(args...; kwargs...), c.callbacks)

function BangBang.push!!(c::MultiCallback{<:Tuple}, callback)
    return MultiCallback((c.callbacks..., callback))
end
function BangBang.push!!(c::MultiCallback{<:AbstractArray}, callback)
    (push!(c.callbacks, callback); return c)
end

"""
    NameFilter(; include=nothing, exclude=nothing)

A filter for variable names.

- If `include` is not `nothing`, only names in `include` will pass the filter.
- If `exclude` is not `nothing`, names in `exclude` will be excluded.
"""
Base.@kwdef struct NameFilter{A,B}
    include::A = nothing
    exclude::B = nothing
end

(f::NameFilter)(name, value) = f(name)
function (f::NameFilter)(name)
    include, exclude = f.include, f.exclude
    return (exclude === nothing || name ∉ exclude) &&
           (include === nothing || name ∈ include)
end

"""
    default_param_names_for_values(x)

Return an iterator of `θ[i]` for each element in `x`.
"""
default_param_names_for_values(x) = ("θ[$i]" for i in 1:length(x))

"""
    params_and_values(model, state; kwargs...)
    params_and_values(model, sampler, state; kwargs...)
    params_and_values(model, transition, state; kwargs...)
    params_and_values(model, sampler, transition, state; kwargs...)

Return an iterator over parameter names and values from a `state` or `transition`.

The default 2-argument generic implementation attempts to call `AbstractMCMC.getparams(state)`.
If you pass a `transition` as the second argument, it will attempt `getparams(transition)`.

To support a specific sampler/model, you should overload:
- `params_and_values(model, ::MyTransitionType)`
- `params_and_values(model, ::MyStateType)`

The 3-argument version `params_and_values(model, transition, state)` attempts to extract
from the `transition` first, and falls back to `state` if the transition yields no parameters.
"""
function params_and_values(model, state; kwargs...)
    try
        # This generic method works for both 'state' and 'transition' objects
        # as long as getparams is defined for them.
        params = getparams(state)
        return zip(default_param_names_for_values(params), params)
    catch
        return ()
    end
end

function params_and_values(model, sampler::AbstractSampler, state; kwargs...)
    return params_and_values(model, state; kwargs...)
end

function params_and_values(model, transition, state; kwargs...)
    # Prioritize transition-based extraction.
    # This calls the generic 2-arg method with the transition object.
    vals = params_and_values(model, transition; kwargs...)

    # If transition extraction returns nothing/empty (e.g., getparams not defined for it),
    # fallback to state-based extraction.
    if isempty(vals)
        return params_and_values(model, state; kwargs...)
    end
    return vals
end

function params_and_values(model, sampler::AbstractSampler, transition, state; kwargs...)
    return params_and_values(model, transition, state; kwargs...)
end

"""
    extras(model, state; kwargs...)
    extras(model, sampler, state; kwargs...)
    extras(model, transition, state; kwargs...)
    extras(model, sampler, transition, state; kwargs...)

Return an iterator with elements of the form `(name, value)` for additional statistics in `state`.

Default implementation uses `AbstractMCMC.getstats(state)` if available and returns
an iterator over the named tuple fields. Returns empty iterator if getstats is not implemented.
"""
function extras(model, state; kwargs...)
    try
        stats = getstats(state)
        if stats isa NamedTuple
            return pairs(stats)
        else
            return ()
        end
    catch
        return ()
    end
end

function extras(model, sampler::AbstractSampler, state; kwargs...)
    return extras(model, state; kwargs...)
end

extras(model, transition, state; kwargs...) = extras(model, state; kwargs...)

function extras(model, sampler::AbstractSampler, transition, state; kwargs...)
    return extras(model, transition, state; kwargs...)
end

"""
    hyperparams(model, sampler[, state]; kwargs...)

Return an iterator with elements of the form `(name, value)` for hyperparameters in `model`.

Default returns an empty iterator. Override for specific model/sampler combinations.
"""
hyperparams(model, sampler; kwargs...) = Pair{String,Any}[]
hyperparams(model, sampler, state; kwargs...) = hyperparams(model, sampler; kwargs...)

"""
    hyperparam_metrics(model, sampler[, state]; kwargs...)

Return a `Vector{String}` of metrics for hyperparameters in `model`.

Default returns an empty vector. Override for specific model/sampler combinations.
"""
hyperparam_metrics(model, sampler; kwargs...) = String[]
function hyperparam_metrics(model, sampler, state; kwargs...)
    return hyperparam_metrics(model, sampler; kwargs...)
end
