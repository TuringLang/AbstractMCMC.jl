struct MyModel <: AbstractMCMC.AbstractModel end

struct MySample{A,B}
    a::A
    b::B
end

struct MySampler <: AbstractMCMC.AbstractSampler end
struct AnotherSampler <: AbstractMCMC.AbstractSampler end

struct MyChain{A,B,S} <: AbstractMCMC.AbstractChains
    as::Vector{A}
    bs::Vector{B}
    stats::S
end

MyChain(a, b) = MyChain(a, b, NamedTuple())

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    state::Union{Nothing,Integer} = nothing;
    sleepy = false,
    loggers = false,
    init_params = nothing,
    kwargs...
)
    # sample `a` is missing in the first step if not provided
    a, b = if state === nothing && init_params !== nothing
        init_params.a, init_params.b
    else
        (state === nothing ? missing : rand(rng)), randn(rng)
    end

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
    stats = nothing,
    kwargs...
)
    as = [t.a for t in samples]
    bs = [t.b for t in samples]

    return MyChain(as, bs, stats)
end

function isdone(
    rng::AbstractRNG,
    model::MyModel,
    s::MySampler,
    samples,
    state,
    iteration::Int;
    kwargs...
)
    # Calculate the mean of x.b.
    bmean = mean(x.b for x in samples)
    return abs(bmean) <= 0.001 || iteration >= 10_000 || state >= 10_000
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
