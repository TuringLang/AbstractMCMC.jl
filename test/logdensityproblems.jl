@testset "logdensityproblems.jl" begin
    # Gaussian log density (without additive constants)
    # Without LogDensityProblems.jl interface
    mylogdensity(x) = -sum(abs2, x) / 2

    # With LogDensityProblems.jl interface
    struct MyLogDensity
        dim::Int
    end
    LogDensityProblems.logdensity(::MyLogDensity, x) = mylogdensity(x)
    LogDensityProblems.dimension(m::MyLogDensity) = m.dim
    LogDensityProblems.capabilities(::Type{MyLogDensity}) = LogDensityProblems.LogDensityOrder{0}()

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

    @testset "LogDensityModel" begin
        ℓ = MyLogDensity(10)
        model = @inferred AbstractMCMC.LogDensityModel(ℓ)
        @test model isa AbstractMCMC.LogDensityModel{MyLogDensity}
        @test model.logdensity === ℓ

        @test_throws ArgumentError AbstractMCMC.LogDensityModel(mylogdensity)
    end

    @testset "fallback for log densities" begin
        # Sample with log density
        dim = 10
        ℓ = MyLogDensity(dim)
        Random.seed!(1234)
        N = 1_000
        samples = sample(ℓ, MySampler(), N)

        # Samples are of the correct dimension and log density values are correct
        @test length(samples) == N
        @test all(length(x.a) == dim for x in samples)
        @test all(x.b ≈ LogDensityProblems.logdensity(ℓ, x.a) for x in samples)

        # Same chain as if LogDensityModel is used explicitly
        Random.seed!(1234)
        samples2 = sample(AbstractMCMC.LogDensityModel(ℓ), MySampler(), N)
        @test length(samples2) == N
        @test all(x.a == y.a && x.b == y.b for (x, y) in zip(samples, samples2))

        # Same chain if sampling is performed with convergence criterion
        Random.seed!(1234)
        isdone(rng, model, sampler, state, samples, iteration; kwargs...) = iteration > N
        samples3 = sample(ℓ, MySampler(), isdone)
        @test length(samples3) == N
        @test all(x.a == y.a && x.b == y.b for (x, y) in zip(samples, samples3))

        # Same chain if sampling is performed with iterator
        Random.seed!(1234)
        samples4 = collect(Iterators.take(AbstractMCMC.steps(ℓ, MySampler()), N))
        @test length(samples4) == N
        @test all(x.a == y.a && x.b == y.b for (x, y) in zip(samples, samples4))

        # Same chain if sampling is performed with transducer
        Random.seed!(1234)
        xf = AbstractMCMC.Sample(ℓ, MySampler())
        samples5 = collect(xf(1:N))
        @test length(samples5) == N
        @test all(x.a == y.a && x.b == y.b for (x, y) in zip(samples, samples5))

        # Parallel sampling
        for alg in (MCMCSerial(), MCMCDistributed(), MCMCThreads())
            chains = sample(ℓ, MySampler(), alg, N, 2)
            @test length(chains) == 2
            samples = vcat(chains[1], chains[2])
            @test length(samples) == 2 * N
            @test all(length(x.a) == dim for x in samples)
            @test all(x.b ≈ LogDensityProblems.logdensity(ℓ, x.a) for x in samples)
        end

        # Log density has to satisfy the LogDensityProblems interface
        @test_throws ArgumentError sample(mylogdensity, MySampler(), N)
        @test_throws ArgumentError sample(mylogdensity, MySampler(), isdone)
        @test_throws ArgumentError sample(mylogdensity, MySampler(), MCMCSerial(), N, 2)
        @test_throws ArgumentError sample(mylogdensity, MySampler(), MCMCThreads(), N, 2)
        @test_throws ArgumentError sample(mylogdensity, MySampler(), MCMCDistributed(), N, 2)
        @test_throws ArgumentError AbstractMCMC.steps(mylogdensity, MySampler())
        @test_throws ArgumentError AbstractMCMC.Sample(mylogdensity, MySampler())
    end
end
