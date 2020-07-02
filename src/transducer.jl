struct Sample{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K} <: Transducers.Transducer
    rng::A
    model::M
    sampler::S
    kwargs::K
end

function Sample(model::AbstractModel, sampler::AbstractSampler; kwargs...)
    return Sample(Random.GLOBAL_RNG, model, sampler; kwargs...)
end

function Sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler;
    kwargs...
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
