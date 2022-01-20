@testset "stepper.jl" begin
    @testset "Iterator sampling" begin
        Random.seed!(1234)
        as = []
        bs = []

        iter = AbstractMCMC.steps(MyModel(), MySampler())
        iter = AbstractMCMC.steps(MyModel(), MySampler(); a = 1.0) # `a` shouldn't do anything

        for (count, t) in enumerate(iter)
            if count >= 1000
                break
            end

            # don't save missing values
            t.a === missing && continue

            push!(as, t.a)
            push!(bs, t.b)
        end

        @test length(as) == length(bs) == 998

        @test mean(as) ≈ 0.5 atol=2e-2
        @test var(as) ≈ 1 / 12 atol=5e-3
        @test mean(bs) ≈ 0.0 atol=5e-2
        @test var(bs) ≈ 1 atol=5e-2

        @test Base.IteratorSize(iter) == Base.IsInfinite()
        @test Base.IteratorEltype(iter) == Base.EltypeUnknown()
    end
end
