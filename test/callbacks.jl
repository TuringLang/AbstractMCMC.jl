@testset "Callbacks" begin
    @testset "MultiCallback" begin
        counts = [Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1

        multi = AbstractMCMC.MultiCallback(cb1, cb2)
        multi(nothing, nothing, nothing, nothing, nothing, 1)

        @test counts[1][] == 1
        @test counts[2][] == 1

        counts = [Ref(0), Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1
        cb3 = (args...; kwargs...) -> counts[3][] += 1

        multi = AbstractMCMC.MultiCallback(cb1, cb2, cb3)
        for _ in 1:10
            multi(nothing, nothing, nothing, nothing, nothing, 1)
        end
        @test all(c[] == 10 for c in counts)
    end

    @testset "MultiCallback dynamic push!!" begin
        using BangBang
        counts = [Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1

        multi = AbstractMCMC.MultiCallback()
        multi = push!!(multi, cb1)
        multi = push!!(multi, cb2)

        for _ in 1:5
            multi(nothing, nothing, nothing, nothing, nothing, 1)
        end
        @test counts[1][] == 5
        @test counts[2][] == 5
    end

    @testset "NameFilter" begin
        f1 = AbstractMCMC.NameFilter(; include=["a", "b"])
        @test f1("a") == true
        @test f1("b") == true
        @test f1("c") == false

        f2 = AbstractMCMC.NameFilter(; exclude=["x", "y"])
        @test f2("a") == true
        @test f2("x") == false
        @test f2("y") == false

        f3 = AbstractMCMC.NameFilter(; include=["a", "b", "c"], exclude=["c"])
        @test f3("a") == true
        @test f3("c") == false
        @test f3("d") == false

        f4 = AbstractMCMC.NameFilter()
        @test f4("anything") == true

        f5 = AbstractMCMC.NameFilter(; include=[:alpha, :beta])
        @test f5(:alpha) == true
        @test f5(:gamma) == false

        f6 = AbstractMCMC.NameFilter(; include=[])
        @test f6("a") == false
        @test f6("b") == false

        f7 = AbstractMCMC.NameFilter(; exclude=[])
        @test f7("a") == true
        @test f7("z") == true
    end

    @testset "MultiCallback with sample" begin
        count = Ref(0)
        cb = (args...; kwargs...) -> count[] += 1

        chain = sample(MyModel(), MySampler(), 100; callback=cb)
        @test count[] == 100

        counts = [Ref(0), Ref(0)]
        multi = AbstractMCMC.MultiCallback(
            (args...; kwargs...) -> counts[1][] += 1,
            (args...; kwargs...) -> counts[2][] += 1,
        )
        chain = sample(MyModel(), MySampler(), 50; callback=multi)
        @test counts[1][] == 50
        @test counts[2][] == 50
    end

    @testset "default_param_names_for_values" begin
        names = collect(AbstractMCMC.default_param_names_for_values([1.0, 2.0, 3.0]))
        @test names == ["θ[1]", "θ[2]", "θ[3]"]

        names = collect(AbstractMCMC.default_param_names_for_values([5.5]))
        @test names == ["θ[1]"]
    end
end

using TensorBoardLogger
using OnlineStats

# Helper to ensure extension is loaded and constants are defined in AbstractMCMC
function ensure_tb_extension_loaded()
    try
        TensorBoardCallback(mktempdir())
    catch
    end
end

@testset "TensorBoardCallback Extension" begin
    ensure_tb_extension_loaded()
    @testset "TensorBoardCallback creation" begin
        logdir = mktempdir()
        cb = TensorBoardCallback(logdir)
        @test typeof(cb).name.name == :TensorBoardCallback
        @test cb.logger isa TensorBoardLogger.TBLogger
    end

    @testset "TensorBoardCallback with custom stats" begin
        logdir = mktempdir()
        custom_stats = Mean()
        cb = TensorBoardCallback(logdir; stats=custom_stats)
        @test typeof(cb).name.name == :TensorBoardCallback
    end

    @testset "TensorBoardCallback with filter options" begin
        logdir = mktempdir()

        cb1 = TensorBoardCallback(logdir; include=["mu", "sigma"])
        @test cb1.variable_filter("mu", 1.0) == true
        @test cb1.variable_filter("other", 1.0) == false

        cb2 = TensorBoardCallback(logdir; exclude=["internal"])
        @test cb2.variable_filter("mu", 1.0) == true
        @test cb2.variable_filter("internal", 1.0) == false

        my_filter = (name, value) -> !startswith(string(name), "x")
        cb3 = TensorBoardCallback(logdir; filter=my_filter)
        @test cb3.variable_filter("mu", 1.0) == true
        @test cb3.variable_filter("x_param", 1.0) == false
    end

    @testset "TensorBoardCallback extras filtering" begin
        logdir = mktempdir()

        cb1 = TensorBoardCallback(logdir; include_extras=true)
        @test cb1.include_extras == true

        cb2 = TensorBoardCallback(logdir; include_extras=false)
        @test cb2.include_extras == false

        cb3 = TensorBoardCallback(logdir; extras_include=["log_density"])
        @test cb3.extras_filter("log_density", 1.0) == true
        @test cb3.extras_filter("acceptance_rate", 1.0) == false

        cb4 = TensorBoardCallback(logdir; extras_exclude=["step_size"])
        @test cb4.extras_filter("log_density", 1.0) == true
        @test cb4.extras_filter("step_size", 1.0) == false
    end

    @testset "TensorBoardCallback hyperparams filtering" begin
        logdir = mktempdir()

        cb1 = TensorBoardCallback(logdir; include_hyperparams=false)
        @test cb1.include_hyperparams == false

        cb2 = TensorBoardCallback(logdir; include_hyperparams=true)
        @test cb2.include_hyperparams == true

        cb3 = TensorBoardCallback(
            logdir; include_hyperparams=true, hyperparams_include=["target_accept"]
        )
        @test cb3.hyperparam_filter("target_accept", 0.8) == true
        @test cb3.hyperparam_filter("other", 1.0) == false
    end

    @testset "TensorBoardCallback prefixes" begin
        logdir = mktempdir()

        cb1 = TensorBoardCallback(logdir; param_prefix="params/")
        @test cb1.param_prefix == "params/"

        cb2 = TensorBoardCallback(logdir; extras_prefix="stats/")
        @test cb2.extras_prefix == "stats/"
    end

    @testset "Skip OnlineStat wrapper" begin
        # Skip(b) skips the first b observations.
        skip = AbstractMCMC.Skip(10, Mean())
        @test skip.b == 10
        @test skip.stat isa Mean

        # Fit 15 items. First 10 (1..10) should be skipped. 11..15 should be fitted.
        for i in 1:15
            OnlineStats.fit!(skip, Float64(i))
        end
        # Mean of 11, 12, 13, 14, 15 is 13.0
        @test OnlineStats.value(skip) ≈ 13.0
        
        skip2 = AbstractMCMC.Skip(5, Variance())
        for i in 1:20
            OnlineStats.fit!(skip2, Float64(i))
        end
        @test OnlineStats.nobs(skip2.stat) == 15
    end

    @testset "Thin OnlineStat wrapper" begin
        # Thin(b) passes every b-th observation.
        thin = AbstractMCMC.Thin(5, Mean())
        @test thin.b == 5
        @test thin.stat isa Mean

        # Fit 1..20. 
        # i=1 (idx=0) -> 0%5==0 -> Fit 1
        # i=6 (idx=5) -> 5%5==0 -> Fit 6
        # i=11 (idx=10) -> 10%5==0 -> Fit 11
        # i=16 (idx=15) -> 15%5==0 -> Fit 16
        for i in 1:20
            OnlineStats.fit!(thin, Float64(i))
        end
        # Mean of 1, 6, 11, 16 = 8.5
        @test OnlineStats.value(thin) ≈ 8.5
    end

    @testset "WindowStat OnlineStat wrapper" begin
        ws = AbstractMCMC.WindowStat(5, Mean())
        @test OnlineStats.nobs(ws) == 0

        # Fill window: 1, 2, 3, 4, 5
        for i in 1:5
            OnlineStats.fit!(ws, Float64(i))
        end
        @test OnlineStats.nobs(ws) == 5
        @test OnlineStats.value(OnlineStats.value(ws)) ≈ 3.0 # Mean(1..5) = 3.0

        # Slide window: 6, 7, 8, 9, 10
        # Window should contain 6, 7, 8, 9, 10
        for i in 6:10
            OnlineStats.fit!(ws, Float64(i))
        end
        @test OnlineStats.nobs(ws) == 5
        stat_result = OnlineStats.value(ws)
        @test stat_result isa Mean
        @test OnlineStats.value(stat_result) ≈ 8.0 # Mean(6..10) = 8.0
        
        # Partially wrap window: 11, 12
        # Buffer should handle wrapping correctly.
        # Window: 8, 9, 10, 11, 12 (in some order internally, properly sorted by value())
        for i in 11:12
            OnlineStats.fit!(ws, Float64(i))
        end
        @test OnlineStats.value(OnlineStats.value(ws)) ≈ 10.0 # Mean(8..12) = 10.0
    end

    @testset "OnlineStat merging" begin
        m1 = Mean()
        m2 = Mean()
        for i in 1:50
            OnlineStats.fit!(m1, Float64(i))
        end
        for i in 51:100
            OnlineStats.fit!(m2, Float64(i))
        end
        merged = OnlineStats.merge!(m1, m2)
        @test OnlineStats.value(merged) ≈ mean(1:100)
    end

    @testset "Series with multiple stats" begin
        series = Series(Mean(), Variance())
        for i in 1:100
            OnlineStats.fit!(series, Float64(i))
        end
        m, v = OnlineStats.value(series)
        @test m ≈ mean(1:100)
        @test OnlineStats.nobs(series) == 100
    end
end
