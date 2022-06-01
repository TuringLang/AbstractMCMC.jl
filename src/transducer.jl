struct Sample{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K} <:
       Transducers.Transducer
    rng::A
    model::M
    sampler::S
    kwargs::K
end

function Sample(model::AbstractModel, sampler::AbstractSampler; kwargs...)
    return Sample(Random.default_rng(), model, sampler; kwargs...)
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

# Initial sample.
function Transducers.start(rf::Transducers.R_{<:Sample}, result)
    # Unpack transducer.
    td = Transducers.xform(rf)
    rng = td.rng
    model = td.model
    sampler = td.sampler
    kwargs = td.kwargs
    discard_initial = get(kwargs, :discard_initial, 0)::Int

    # Start sampling algorithm and discard initial samples if desired.
    sample, state = step(rng, model, sampler; kwargs...)
    for _ in 1:discard_initial
        sample, state = step(rng, model, sampler, state; kwargs...)
    end

    return Transducers.wrap(
        rf, (sample, state), Transducers.start(Transducers.inner(rf), result)
    )
end

# Subsequent samples.
function Transducers.next(rf::Transducers.R_{<:Sample}, result, input)
    # Unpack transducer.
    td = Transducers.xform(rf)
    rng = td.rng
    model = td.model
    sampler = td.sampler
    kwargs = td.kwargs
    thinning = get(kwargs, :thinning, 1)::Int

    let rng = rng,
        model = model,
        sampler = sampler,
        kwargs = kwargs,
        thinning = thinning,
        inner_rf = Transducers.inner(rf)

        Transducers.wrapping(rf, result) do (sample, state), iresult
            iresult2 = Transducers.next(inner_rf, iresult, sample)

            # Perform thinning if desired.
            for _ in 1:(thinning - 1)
                _, state = step(rng, model, sampler, state; kwargs...)
            end

            return step(rng, model, sampler, state; kwargs...), iresult2
        end
    end
end

function Transducers.complete(rf::Transducers.R_{Sample}, result)
    _, inner_result = Transducers.unwrap(rf, result)
    return Transducers.complete(Transducers.inner(rf), inner_result)
end
