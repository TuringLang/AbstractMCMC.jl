@testset "stepper.jl" begin
    @testset "Iterator sampling" begin
        Random.seed!(1234)
        as = []
        bs = []

        iter = AbstractMCMC.steps(MyModel(), MySampler())
        iter = AbstractMCMC.steps(MyModel(), MySampler(); a=1.0) # `a` shouldn't do anything

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

        @test mean(as) ≈ 0.5 atol = 2e-2
        @test var(as) ≈ 1 / 12 atol = 5e-3
        @test mean(bs) ≈ 0.0 atol = 5e-2
        @test var(bs) ≈ 1 atol = 5e-2

        @test Base.IteratorSize(iter) == Base.IsInfinite()
        @test Base.IteratorEltype(iter) == Base.EltypeUnknown()
    end

    @testset "Discard initial samples" begin
        # Create a chain of `N` samples after discarding some initial samples.
        Random.seed!(1234)
        N = 50
        discard_initial = 10
        iter = AbstractMCMC.steps(MyModel(), MySampler(); discard_initial=discard_initial)
        as = []
        bs = []
        for t in Iterators.take(iter, N)
            push!(as, t.a)
            push!(bs, t.b)
        end

        # Repeat sampling with `sample`.
        Random.seed!(1234)
        chain = sample(
            MyModel(), MySampler(), N; discard_initial=discard_initial, progress=false
        )
        @test all(as[i] === chain[i].a for i in 1:N)
        @test all(bs[i] === chain[i].b for i in 1:N)
    end

    @testset "Thin chain by a factor of `thinning`" begin
        # Create a thinned chain with a thinning factor of `thinning`.
        Random.seed!(1234)
        N = 50
        thinning = 3
        iter = AbstractMCMC.steps(MyModel(), MySampler(); thinning=thinning)
        as = []
        bs = []
        for t in Iterators.take(iter, N)
            push!(as, t.a)
            push!(bs, t.b)
        end

        # Repeat sampling with `sample`.
        Random.seed!(1234)
        chain = sample(MyModel(), MySampler(), N; thinning=thinning, progress=false)
        @test all(as[i] === chain[i].a for i in 1:N)
        @test all(bs[i] === chain[i].b for i in 1:N)
    end
end
