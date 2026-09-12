module SamplingOutputTests

using AbstractMCMC
using Random
using Test

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
    samples = reshape([(x=1,), (x=2,)], :, 1)
    chain = SamplingOutput(samples)
    @test chain[2, 1] == (x=2,)
    @test AbstractMCMC.to_samples(NamedTuple, chain) === samples
    @test AbstractMCMC.from_samples(SamplingOutput, samples).samples === samples
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
    @test_throws ArgumentError AbstractMCMC.chainscat(
        chain, SamplingOutput(samples; iterations=2:3)
    )

    converted =
        sample(
            Xoshiro(1),
            Model(),
            Sampler(),
            3;
            chain_type=ConvertedChain,
            discard_initial=2,
            thinning=3,
            save_state=true,
            progress=false,
        ).chain
    @test converted.iterations == 3:3:9
    @test converted.sampler_states == [9]
    @test only(converted.sampling_stats).duration >= 0
    for chain_type in (Any, Vector{NamedTuple})
        raw = sample(Xoshiro(1), Model(), Sampler(), 3; chain_type, progress=false)
        @test raw isa Vector{<:NamedTuple}
    end

    matrix = [(x=i, walker=j) for i in 1:3, j in 1:2]
    for draws in (matrix, view(matrix, :, :))
        chain = AbstractMCMC.bundle_samples(
            draws,
            Model(),
            Sampler(),
            9,
            SamplingOutput;
            discard_initial=2,
            thinning=3,
            save_state=true,
        )
        @test chain.samples == matrix
        @test chain.iterations == 3:3:9
        @test chain.sampler_states == [9, 9]
        @test all(ismissing, chain.sampling_stats)
        draws === matrix && @test chain.samples === matrix
    end

    @testset "sampling: $ensemble" for ensemble in (nothing, MCMCSerial(), MCMCThreads())
        args = ensemble === nothing ? (3,) : (ensemble, 3, 2)
        chain = sample(
            Xoshiro(1),
            Model(),
            Sampler(),
            args...;
            chain_type=SamplingOutput,
            discard_initial=2,
            thinning=3,
            save_state=true,
            progress=false,
        )
        @test size(chain) == (3, ensemble === nothing ? 1 : 2)
        @test chain.iterations == 3:3:9
        @test all(==(9), chain.sampler_states)
        @test all(s -> s.duration >= 0, chain.sampling_stats)
        @test all(c -> getproperty.(c, :iteration) == 3:3:9, eachcol(chain.samples))
    end
end

end
