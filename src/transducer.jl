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
    sample_init!(rng, model, sampler, 0)
    return Sample(rng, model, sampler, kwargs)
end

function Transducers.start(rf::Transducers.R_{<:Sample}, result)
    return Transducers.wrap(rf, nothing, Transducers.start(Transducers.inner(rf), result))
end

function Transducers.next(rf::Transducers.R_{<:Sample}, result, input)
    t = Transducers.xform(rf)
    Transducers.wrapping(rf, result) do state, iresult
        transition = step!(t.rng, t.model, t.sampler, 1, state; t.kwargs...)
        iinput = transition
        iresult = Transducers.next(Transducers.inner(rf), iresult, transition)
        return transition, iresult
    end
end

function Transducers.complete(rf::Transducers.R_{Sample}, result)
    _private_state, inner_result = Transducers.unwrap(rf, result)
    return Transducers.complete(Transducers.inner(rf), inner_result)
end
