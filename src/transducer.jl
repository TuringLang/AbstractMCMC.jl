struct Sample{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K} <:
       Transducers.Transducer
    rng::A
    model::M
    sampler::S
    kwargs::K
end

function Sample(model::AbstractModel, sampler::AbstractSampler; kwargs...)
    return Sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

"""
    Sample([rng, ]model, sampler; kwargs...)

Create a transducer that returns samples from the `model` with the Markov chain Monte Carlo
`sampler`.

# Examples

```jldoctest; setup=:(using AbstractMCMC: Sample)
julia> struct MyModel <: AbstractMCMC.AbstractModel end

julia> struct MySampler <: AbstractMCMC.AbstractSampler end

julia> function AbstractMCMC.step(rng, ::MyModel, ::MySampler, state=nothing; kwargs...)
           # all samples are zero
           return 0.0, state
       end

julia> transducer = Sample(MyModel(), MySampler());

julia> collect(transducer(1:10)) == zeros(10)
true
```
"""
function Sample(
    rng::Random.AbstractRNG, model::AbstractModel, sampler::AbstractSampler; kwargs...
)
    return Sample(rng, model, sampler, kwargs)
end

function Transducers.start(rf::Transducers.R_{<:Sample}, result)
    sampler = Transducers.xform(rf)
    return Transducers.wrap(
        rf,
        step(sampler.rng, sampler.model, sampler.sampler; sampler.kwargs...),
        Transducers.start(Transducers.inner(rf), result),
    )
end

function Transducers.next(rf::Transducers.R_{<:Sample}, result, input)
    t = Transducers.xform(rf)
    Transducers.wrapping(rf, result) do (sample, state), iresult
        iresult2 = Transducers.next(Transducers.inner(rf), iresult, sample)
        return step(t.rng, t.model, t.sampler, state; t.kwargs...), iresult2
    end
end

function Transducers.complete(rf::Transducers.R_{Sample}, result)
    _private_state, inner_result = Transducers.unwrap(rf, result)
    return Transducers.complete(Transducers.inner(rf), inner_result)
end
