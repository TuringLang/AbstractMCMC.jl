struct Stepper{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K}
    rng::A
    model::M
    sampler::S
    kwargs::K
end

Base.iterate(stp::Stepper) = step(stp.rng, stp.model, stp.sampler; stp.kwargs...)
function Base.iterate(stp::Stepper, state)
    return step(stp.rng, stp.model, stp.sampler, state; stp.kwargs...)
end

Base.IteratorSize(::Type{<:Stepper}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:Stepper}) = Base.EltypeUnknown()

"""
    steps([rng::AbstractRNG, ]model::AbstractModel, s::AbstractSampler, kwargs...)

Return an iterator that returns samples continuously.

# Examples

```julia
for transition in steps(MyModel(), MySampler())
    println(transition)

    # Do other stuff with transition below.
end
```
"""
function steps(
    model::AbstractModel,
    sampler::AbstractSampler,
    kwargs...
)
    return steps(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

function steps(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    kwargs...
)
    return Stepper(rng, model, sampler, kwargs)
end
