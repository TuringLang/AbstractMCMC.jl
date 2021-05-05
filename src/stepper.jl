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

function steps(
    model::AbstractModel,
    sampler::AbstractSampler;
    kwargs...
)
    return steps(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

"""
    steps([rng, ]model, sampler; kwargs...)

Create an iterator that returns samples from the `model` with the Markov chain Monte Carlo
`sampler`.

# Examples

```jldoctest; setup=:(using AbstractMCMC: steps)
julia> struct MyModel <: AbstractMCMC.AbstractModel end

julia> struct MySampler <: AbstractMCMC.AbstractSampler end

julia> function AbstractMCMC.step(rng, ::MyModel, ::MySampler, state=nothing; kwargs...)
           # all samples are zero
           return 0.0, state
       end

julia> iterator = steps(MyModel(), MySampler());

julia> collect(Iterators.take(iterator, 10)) == zeros(10)
true
```
"""
function steps(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler;
    kwargs...
)
    return Stepper(rng, model, sampler, kwargs)
end
