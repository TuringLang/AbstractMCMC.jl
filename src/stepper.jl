struct Stepper{A<:Random.AbstractRNG,M<:AbstractModel,S<:AbstractSampler,K}
    rng::A
    model::M
    sampler::S
    kwargs::K
end

# Initial sample.
function Base.iterate(stp::Stepper)
    # Unpack iterator.
    rng = stp.rng
    model = stp.model
    sampler = stp.sampler
    kwargs = stp.kwargs
    discard_initial = get(kwargs, :discard_initial, 0)::Int

    # Start sampling algorithm and discard initial samples if desired.
    sample, state = step(rng, model, sampler; kwargs...)
    for _ in 1:discard_initial
        sample, state = step(rng, model, sampler, state; kwargs...)
    end
    return sample, state
end

# Subsequent samples.
function Base.iterate(stp::Stepper, state)
    # Unpack iterator.
    rng = stp.rng
    model = stp.model
    sampler = stp.sampler
    kwargs = stp.kwargs
    thinning = get(kwargs, :thinning, 1)::Int

    # Return next sample, possibly after thinning the chain if desired.
    for _ in 1:(thinning - 1)
        _, state = step(rng, model, sampler, state; kwargs...)
    end
    return step(rng, model, sampler, state; kwargs...)
end

Base.IteratorSize(::Type{<:Stepper}) = Base.IsInfinite()
Base.IteratorEltype(::Type{<:Stepper}) = Base.EltypeUnknown()

function steps(model_or_logdensity, sampler::AbstractSampler; kwargs...)
    return steps(Random.default_rng(), model_or_logdensity, sampler; kwargs...)
end

"""
    steps(
        rng::Random.AbstractRNG=Random.default_rng(),
        model::AbstractModel,
        sampler::AbstractSampler;
        kwargs...,
    )

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
    rng::Random.AbstractRNG, model::AbstractModel, sampler::AbstractSampler; kwargs...
)
    return Stepper(rng, model, sampler, kwargs)
end

"""
    steps(
        rng::Random.AbstractRNG=Random.default_rng(),
        logdensity,
        sampler::AbstractSampler;
        kwargs...,
    )

Wrap the `logdensity` function in a [`LogDensityModel`](@ref), and call `steps` with the resulting model instead of `logdensity`.

The `logdensity` function has to support the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
function steps(rng::Random.AbstractRNG, logdensity, sampler::AbstractSampler; kwargs...)
    return steps(rng, _model(logdensity), sampler; kwargs...)
end
