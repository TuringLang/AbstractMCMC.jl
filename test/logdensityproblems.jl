@testset "logdensityproblems.jl" begin
    # Add worker processes.
    # Memory requirements on Windows are ~4x larger than on Linux, hence number of processes is reduced
    # See, e.g., https://github.com/JuliaLang/julia/issues/40766 and https://github.com/JuliaLang/Pkg.jl/pull/2366
    pids = addprocs(Sys.iswindows() ? div(Sys.CPU_THREADS::Int, 2) : Sys.CPU_THREADS::Int)

    # Load all required packages (`utils.jl` needs LogDensityProblems, Logging, and Random).
    @everywhere begin
        using AbstractMCMC
        using AbstractMCMC: sample
        using LogDensityProblems

        using Logging
        using Random
        include("utils.jl")
    end

    @testset "LogDensityModel" begin
        ℓ = MyLogDensity(10)
        model = @inferred AbstractMCMC.LogDensityModel(ℓ)
        @test model isa AbstractMCMC.LogDensityModel{MyLogDensity}
        @test model.logdensity === ℓ

        @test_throws ArgumentError AbstractMCMC.LogDensityModel(mylogdensity)

        try
            LogDensityProblems.logdensity(model, ones(10))
        catch exc
            @test exc isa MethodError
            if isdefined(Base.Experimental, :register_error_hint)
                @test occursin("is a wrapper", sprint(showerror, exc))
            end
        end
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
        @test_throws ArgumentError sample(
            mylogdensity, MySampler(), MCMCDistributed(), N, 2
        )
        @test_throws ArgumentError AbstractMCMC.steps(mylogdensity, MySampler())
        @test_throws ArgumentError AbstractMCMC.Sample(mylogdensity, MySampler())
    end

    # Remove workers
    rmprocs(pids...)
end
