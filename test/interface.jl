struct MyModel <: AbstractMCMC.AbstractModel end

struct MyTransition{A,B}
    a::A
    b::B
end

struct MySampler <: AbstractMCMC.AbstractSampler end
struct AnotherSampler <: AbstractMCMC.AbstractSampler end

struct MyChain{A,B} <: AbstractMCMC.AbstractChains
    as::Vector{A}
    bs::Vector{B}
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    N::Integer,
    transition::Union{Nothing,MyTransition};
    sleepy = false,
    loggers = false,
    kwargs...
)
    # sample `a` is missing in the first step
    a = transition === nothing ? missing : rand(rng)
    b = randn(rng)

    loggers && push!(LOGGERS, Logging.current_logger())
    sleepy && sleep(0.001)

    return MyTransition(a, b)
end

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    N::Integer,
    transitions::Vector{<:MyTransition},
    chain_type::Type{MyChain};
    kwargs...
)
    as = [t.a for t in transitions]
    bs = [t.b for t in transitions]

    return MyChain(as, bs)
end

function is_done(
    rng::AbstractRNG,
    model::MyModel,
    s::MySampler,
    transitions,
    iteration::Int;
    chain_type::Type=Any,
    kwargs...
)
    # Calculate the mean of x.b.
    bmean = mean(x.b for x in transitions)
    return abs(bmean) <= 0.001 || iteration >= 10_000
end

# Set a default convergence function.
AbstractMCMC.sample(model, sampler::MySampler; kwargs...) = sample(Random.GLOBAL_RNG, model, sampler, is_done; kwargs...)
AbstractMCMC.chainscat(chains::Union{MyChain,Vector{<:MyChain}}...) = vcat(chains...)
