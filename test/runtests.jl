using AbstractMCMC

using Random
using Statistics
using Test

include("interface.jl")

@testset "AbstractMCMC" begin
    @testset "Basic sampling" begin
        Random.seed!(1234)
        N = 1_000
        chain = sample(MyModel(), MySampler(), N; progress = true, sleepy = true)

        # test output type and size
        @test chain isa MyChain
        @test length(chain.as) == N
        @test length(chain.bs) == N

        # test some statistical properties
        @test mean(chain.as) ≈ 0.5 atol=1e-2
        @test var(chain.as) ≈ 1 / 12 atol=5e-3
        @test mean(chain.bs) ≈ 0.0 atol=5e-2
        @test var(chain.bs) ≈ 1 atol=5e-2
    end

    if VERSION ≥ v"1.3"
        @testset "Parallel sampling" begin
            println("testing parallel sampling with ", Threads.nthreads(), " thread(s)...")

            Random.seed!(1234)
            chains = psample(MyModel(), MySampler(), 10_000, 1_000)

            # test output type and size
            @test chains isa Vector{MyChain}
            @test length(chains) == 1000
            @test all(x -> length(x.as) == length(x.bs) == 10_000, chains)

            # test some statistical properties
            @test all(x -> isapprox(mean(x.as), 0.5; atol=1e-2), chains)
            @test all(x -> isapprox(var(x.as), 1 / 12; atol=5e-3), chains)
            @test all(x -> isapprox(mean(x.bs), 0; atol=5e-2), chains)
            @test all(x -> isapprox(var(x.bs), 1; atol=5e-2), chains)

            # test reproducibility
            Random.seed!(1234)
            chains2 = psample(MyModel(), MySampler(), 10_000, 1000)

            @test all(((x, y),) -> x.as == y.as && x.bs == y.bs, zip(chains, chains2))
        end
    end

    @testset "Chain constructors" begin
        chain1 = sample(MyModel(), MySampler(), 100; progress = true, sleepy = true)
        chain2 = sample(MyModel(), MySampler(), 100; progress = true, sleepy = true, chain_type = Vector)

        @test chain1 isa MyChain
        @test chain2 isa Vector{MyTransition}
    end
end