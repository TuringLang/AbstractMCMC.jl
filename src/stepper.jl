struct Stepper{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K}
    rng::A
    model::M
    s::S
    kwargs::K
end

function Base.iterate(stp::Stepper, state=nothing)
    t = step!(stp.rng, stp.model, stp.s, 1, state; stp.kwargs...)
    return t, t
end

Base.IteratorSize(::Type{<:Stepper}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:Stepper}) = Base.EltypeUnknown()

"""
    steps!([rng::AbstractRNG, ]model::AbstractModel, s::AbstractSampler, kwargs...)

Return an iterator that returns samples continuously.

# Examples

```julia
for transition in steps!(MyModel(), MySampler())
    println(transition)

    # Do other stuff with transition below.
end
```
"""
function steps!(
    model::AbstractModel,
    s::AbstractSampler,
    kwargs...
)
    return steps!(Random.GLOBAL_RNG, model, s; kwargs...)
end

function steps!(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    s::AbstractSampler,
    kwargs...
)
    return Stepper(rng, model, s, kwargs)
end
