module SamplingOutputTests
using AbstractMCMC, Random, Test
using AbstractMCMC: bundle_samples, chainscat, chainsstack, from_samples, to_samples

struct Model <: AbstractMCMC.AbstractModel end
struct Sampler <: AbstractMCMC.AbstractSampler end
function AbstractMCMC.step(rng, ::Model, ::Sampler, state=0; kwargs...)
    return (x=rand(rng), iteration=state + 1), state + 1
end

struct ConvertedChain{C<:SamplingOutput} <: AbstractMCMC.AbstractChains
    chain::C
end
Base.convert(::Type{ConvertedChain}, chain::SamplingOutput) = ConvertedChain(chain)

@testset "SamplingOutput" begin
    opts = (; discard_initial=2, thinning=3, save_state=true, progress=false)
    samples = reshape([(x=1,), (x=2,)], :, 1)
    chain = SamplingOutput(samples)
    @test chain[2, 1] == (x=2,)
    @test to_samples(eltype(samples), chain) === samples
    @test to_samples(NamedTuple, chain) isa Matrix{NamedTuple}
    @test to_samples(NamedTuple, chain) == samples
    @test to_samples(Float64, SamplingOutput(ones(Int, 1, 1))) isa Matrix{Float64}
    @test from_samples(SamplingOutput, samples).samples === samples
    @test from_samples(typeof(chain), to_samples(NamedTuple, chain)) isa typeof(chain)
    @test from_samples(typeof(chain), samples).samples === samples
    @test from_samples(SamplingOutput{Float64}, ones(Int, 1, 1)) isa SamplingOutput{Float64}
    many = [SamplingOutput(fill(i, 128, 1)) for i in 1:64]
    @test chainsstack(many).samples == repeat(reshape(1:64, 1, :), 128)
    @test (@allocated chainsstack(many)) < 4 * 128 * 64 * sizeof(Int)
    matrix = [(x=i, walker=j) for i in 1:3, j in 1:2]
    indexed = SamplingOutput(matrix)
    @test indexed[begin] == matrix[begin]
    @test indexed[end] == matrix[end]
    @test indexed[begin, end] == matrix[begin, end]
    @test indexed[end, begin] == matrix[end, begin]
    @test indexed[begin:end, begin:end] == matrix
    @test convert(SamplingOutput, chain) === chain
    @test convert(typeof(chain), chain) === chain
    @test all(ismissing, chain.sampling_stats)
    @test all(ismissing, chain.sampler_states)
    plain = sprint(show, MIME"text/plain"(), chain)
    @test startswith(plain, "SamplingOutput: 1 chain, 2 draws each\n  Total draws:")
    @test occursin("use FlexiChains, MCMCChains, or ArviZ.", plain)
    @test occursin("Total draws:    2 (discard_initial = 0, thinning = 1)", plain)
    @test !occursin('\e', plain)
    @test occursin('\e', sprint(show, MIME"text/plain"(), chain; context=:color => true))
    @test_throws DimensionMismatch SamplingOutput(samples; iterations=1:3)
    @test_throws ArgumentError SamplingOutput(samples; iterations=2:-1:1)
    @test_throws DimensionMismatch SamplingOutput(
        samples; sampling_stats=[missing, missing]
    )
    @test_throws DimensionMismatch SamplingOutput(samples; sampler_states=[])
    @test_throws ArgumentError chainscat(chain, SamplingOutput(samples; iterations=2:3))

    converted = sample(
        Xoshiro(1), Model(), Sampler(), 3; chain_type=ConvertedChain, opts...
    )
    converted = converted.chain
    @test converted.iterations == 3:3:9
    @test converted.sampler_states == [9]
    @test only(converted.sampling_stats).duration >= 0
    @test_throws ArgumentError from_samples(typeof(converted), converted.samples)
    for chain_type in (Any, Vector{NamedTuple})
        raw = sample(Xoshiro(1), Model(), Sampler(), 3; chain_type, progress=false)
        @test raw isa Vector{<:NamedTuple}
    end
    for draws in (matrix, view(matrix, :, :))
        chain = bundle_samples(draws, Model(), Sampler(), 9, SamplingOutput; opts...)
        @test chain.samples == matrix
        @test chain.iterations == 3:3:9
        @test chain.sampler_states == [9, 9]
        @test all(ismissing, chain.sampling_stats)
        draws === matrix && @test chain.samples === matrix
    end
    @testset "sampling: $ensemble" for ensemble in (nothing, MCMCSerial(), MCMCThreads())
        args = ensemble === nothing ? (3,) : (ensemble, 3, 2)
        chain = sample(
            Xoshiro(1), Model(), Sampler(), args...; chain_type=SamplingOutput, opts...
        )
        @test size(chain) == (3, ensemble === nothing ? 1 : 2)
        @test chain.iterations == 3:3:9
        @test all(==(9), chain.sampler_states)
        @test all(s -> s.duration >= 0, chain.sampling_stats)
        @test all(c -> getproperty.(c, :iteration) == 3:3:9, eachcol(chain.samples))
    end
end
end
