struct MyModel <: AbstractMCMC.AbstractModel end

struct MySample{A,B}
    a::A
    b::B
    is_warmup::Bool
end

MySample(a, b) = MySample(a, b, false)

struct MySampler <: AbstractMCMC.AbstractSampler end
struct AnotherSampler <: AbstractMCMC.AbstractSampler end

struct MyChain{A,B,S} <: AbstractMCMC.AbstractChains
    as::Vector{A}
    bs::Vector{B}
    stats::S
end

MyChain(a, b) = MyChain(a, b, NamedTuple())

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    state::Union{Nothing,Integer}=nothing;
    loggers=false,
    initial_params=nothing,
    num_warmup,
    kwargs...,
)
    num_warmup isa Integer ||
        error("num_warmup should have been passed as a keyword argument to step_warmup")
    transition, state = AbstractMCMC.step(
        rng, model, sampler, state; loggers, initial_params, kwargs...
    )
    return MySample(transition.a, transition.b, true), state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    state::Union{Nothing,Integer}=nothing;
    loggers=false,
    initial_params=nothing,
    kwargs...,
)
    # sample `a` is missing in the first step if not provided
    a, b = if state === nothing && initial_params !== nothing
        initial_params.a, initial_params.b
    else
        (state === nothing ? missing : rand(rng)), randn(rng)
    end

    loggers && push!(LOGGERS, Logging.current_logger())

    _state = state === nothing ? 1 : state + 1

    return MySample(a, b), _state
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:MySample},
    model::MyModel,
    sampler::MySampler,
    ::Any,
    ::Type{MyChain};
    stats=nothing,
    kwargs...,
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
    kwargs...,
)
    # Calculate the mean of x.b.
    bmean = mean(x.b for x in samples)
    return abs(bmean) <= 0.001 || iteration > 10_000
end

# Set a default convergence function.
function AbstractMCMC.sample(model, sampler::MySampler; kwargs...)
    return sample(Random.default_rng(), model, sampler, isdone; kwargs...)
end

function AbstractMCMC.chainscat(
    chain::Union{MyChain,Vector{<:MyChain}}, chains::Union{MyChain,Vector{<:MyChain}}...
)
    return vcat(chain, chains...)
end

# Conversion to NamedTuple
Base.convert(::Type{NamedTuple}, x::MySample) = (a=x.a, b=x.b)

# Gaussian log density (without additive constants)
# Without LogDensityProblems.jl interface
mylogdensity(x) = -sum(abs2, x) / 2

# With LogDensityProblems.jl interface
struct MyLogDensity
    dim::Int
end
LogDensityProblems.logdensity(::MyLogDensity, x) = mylogdensity(x)
LogDensityProblems.dimension(m::MyLogDensity) = m.dim
function LogDensityProblems.capabilities(::Type{MyLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

# Define "sampling"
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.LogDensityModel{MyLogDensity},
    ::MySampler,
    state::Union{Nothing,Integer}=nothing;
    kwargs...,
)
    # Sample from multivariate normal distribution    
    ℓ = model.logdensity
    dim = LogDensityProblems.dimension(ℓ)
    θ = randn(rng, dim)
    logdensity_θ = LogDensityProblems.logdensity(ℓ, θ)

    _state = state === nothing ? 1 : state + 1
    return MySample(θ, logdensity_θ), _state
end

AbstractMCMC.getparams(state::Integer) = Float64[]
AbstractMCMC.getstats(state::Integer) = (iteration=state,)
