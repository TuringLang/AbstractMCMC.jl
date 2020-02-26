struct MyModel <: AbstractMCMC.AbstractModel end

struct MyTransition
    a::Float64
    b::Float64
end

struct MySampler <: AbstractMCMC.AbstractSampler end

struct MyChain <: AbstractMCMC.AbstractChains
    as::Vector{Float64}
    bs::Vector{Float64}
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    N::Integer,
    transition::Union{Nothing,MyTransition};
    sleepy = false,
    kwargs...
)
    a = rand(rng)
    b = randn(rng)

    sleepy && sleep(0.001)

    return MyTransition(a, b)
end

function AbstractMCMC.bundle_samples(
    rng::AbstractRNG,
    model::MyModel,
    sampler::MySampler,
    N::Integer,
    transitions::Vector{MyTransition},
    chain_type::Type{MyChain};
    kwargs...
)
    n = length(transitions)
    as = Vector{Float64}(undef, n)
    bs = Vector{Float64}(undef, n)
    for i in 1:n
        transition = transitions[i]
        as[i] = transition.a
        bs[i] = transition.b
    end

    return MyChain(as, bs)
end

function AbstractMCMC.done_sampling(
    rng::AbstractRNG,
    model::MyModel,
    s::MySampler,
    transitions,
    iteration::Int;
    chain_type::Type=Any,
    kwargs...
)
    # Calculate the mean of x.b.
    bmean = mean(map(x -> x.b, transitions))

    return isapprox(bmean, 0.0, atol=0.001) || iteration >= 10_000
end

AbstractMCMC.chainscat(chains::Union{MyChain,Vector{<:MyChain}}...) = vcat(chains...)