@testset "transducer.jl" begin
    Random.seed!(1234)

    @testset "Basic sampling" begin
        N = 1_000
        local chain
        Logging.with_logger(TerminalLogger()) do
            xf = AbstractMCMC.Sample(MyModel(), MySampler();
                                     sleepy = true, logger = true)
            chain = collect(xf, withprogress(1:N; interval=1e-3))
        end

        # test output type and size
        @test chain isa Vector{<:MyTransition}
        @test length(chain) == N

        # test some statistical properties
        tail_chain = @view chain[2:end]
        @test mean(x.a for x in tail_chain) ≈ 0.5 atol=6e-2
        @test var(x.a for x in tail_chain) ≈ 1 / 12 atol=5e-3
        @test mean(x.b for x in tail_chain) ≈ 0.0 atol=5e-2
        @test var(x.b for x in tail_chain) ≈ 1 atol=6e-2
    end

    @testset "drop" begin
        xf = AbstractMCMC.Sample(MyModel(), MySampler())
        chain = collect(xf |> Drop(1), 1:10)
        @test chain isa Vector{MyTransition{Float64,Float64}}
        @test length(chain) == 9
    end

    # Reproduce iterator example
    @testset "iterator example" begin
        # filter missing values and split transitions
        xf = AbstractMCMC.Sample(MyModel(), MySampler()) |>
            OfType(MyTransition{Float64,Float64}) |> Map(x -> (x.a, x.b))
        as, bs = foldl(xf, 1:999; init = (Float64[], Float64[])) do (as, bs), (a, b)
            push!(as, a)
            push!(bs, b)
            as, bs
        end

        @test length(as) == length(bs) == 998

        @test mean(as) ≈ 0.5 atol=1e-2
        @test var(as) ≈ 1 / 12 atol=5e-3
        @test mean(bs) ≈ 0.0 atol=5e-2
        @test var(bs) ≈ 1 atol=5e-2
    end
end
