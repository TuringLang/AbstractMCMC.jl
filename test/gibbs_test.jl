using Test
using Random
using AbstractMCMC
using Turing
using Statistics

@testset "AbstractMCMC Gibbs combinator" begin
    @model function normal_model(x)
        μ ~ Normal(0.0, 10.0)
        σ ~ truncated(Normal(0.0, 5.0), 0.0, Inf)
        for i in eachindex(x)
            x[i] ~ Normal(μ, σ)
        end
    end

    rng = MersenneTwister(42)
    true_μ = 3.0
    true_σ = 1.5
    x_obs = true_μ .+ true_σ .* randn(rng, 50)
    model = normal_model(x_obs)

    @testset "Gibbs(μ=>MH, σ=>MH) recovers posterior" begin
        spl = AbstractMCMC.Gibbs(@varname(μ) => MH(), @varname(σ) => MH())
        chain = sample(rng, model, spl, 1000; progress=false)
        @test abs(mean(chain[:μ]) - mean(x_obs)) < 0.5
        @test abs(mean(chain[:σ]) - true_σ) < 0.5
    end

    @testset "Gibbs is AbstractSampler" begin
        spl = AbstractMCMC.Gibbs(@varname(μ) => MH(), @varname(σ) => MH())
        @test spl isa AbstractMCMC.AbstractSampler
    end

    @testset "GibbsState has correct structure" begin
        spl = AbstractMCMC.Gibbs(@varname(μ) => MH(), @varname(σ) => MH())
        _, state = AbstractMCMC.step(rng, model, spl)
        @test state isa AbstractMCMC.GibbsState
        @test length(state.sub_states) == 2
    end

    @testset "Mismatched varnames/samplers raises ArgumentError" begin
        @test_throws ArgumentError AbstractMCMC.Gibbs(
            ([@varname(μ), @varname(σ)],),
            (MH(), MH()),
        )
    end

    @testset "Existing Turing.Inference.Gibbs still works (no regression)" begin
        chain = sample(rng, model, Turing.Inference.Gibbs(@varname(μ) => MH(), @varname(σ) => MH()), 500; progress=false)
        @test abs(mean(chain[:μ]) - mean(x_obs)) < 0.5
    end
end
