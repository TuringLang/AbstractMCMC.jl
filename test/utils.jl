struct MyModel <: AbstractMCMC.AbstractModel end

struct MySample{A,B}
    a::A
    b::B
end

struct MySampler <: AbstractMCMC.AbstractSampler end
struct AnotherSampler <: AbstractMCMC.AbstractSampler end

struct MyChain{A,B} <: AbstractMCMC.AbstractChains
    as::Vector{A}
    bs::Vector{B}
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    state::Union{Nothing,Integer} = nothing;
    sleepy = false,
    loggers = false,
    kwargs...
)
    # sample `a` is missing in the first step
    a = state === nothing ? missing : rand(rng)
    b = randn(rng)

    loggers && push!(LOGGERS, Logging.current_logger())
    sleepy && sleep(0.001)

    _state = state === nothing ? 1 : state + 1

    return MySample(a, b), _state
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:MySample},
    model::MyModel,
    sampler::MySampler,
    ::Any,
    ::Type{MyChain};
    kwargs...
)
    as = [t.a for t in samples]
    bs = [t.b for t in samples]

    return MyChain(as, bs)
end

function isdone(
    rng::AbstractRNG,
    model::MyModel,
    s::MySampler,
    samples,
    iteration::Int;
    kwargs...
)
    # Calculate the mean of x.b.
    bmean = mean(x.b for x in samples)
    return abs(bmean) <= 0.001 || iteration >= 10_000
end

# Set a default convergence function.
function AbstractMCMC.sample(model, sampler::MySampler; kwargs...)
    return sample(Random.GLOBAL_RNG, model, sampler, isdone; kwargs...)
end

function AbstractMCMC.chainscat(
    chain::Union{MyChain,Vector{<:MyChain}},
    chains::Union{MyChain,Vector{<:MyChain}}...
)
    return vcat(chain, chains...)
end

# Conversion to NamedTuple
Base.convert(::Type{NamedTuple}, x::MySample) = (a = x.a, b = x.b)
