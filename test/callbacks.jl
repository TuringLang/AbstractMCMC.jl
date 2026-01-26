####################################
### Basic Callback Functionality ###
####################################

@testset "Basic mcmc_callback" begin
    @testset "Function callback" begin
        count = Ref(0)
        cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
            count[] += 1
        end

        @test cb isa AbstractMCMC.MultiCallback
        chain = sample(MyModel(), MySampler(), 100; callback=cb)
        @test count[] == 100
    end

    @testset "Multiple function callbacks" begin
        counts = [Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1

        cb = mcmc_callback(cb1, cb2)
        @test cb isa AbstractMCMC.MultiCallback

        chain = sample(MyModel(), MySampler(), 50; callback=cb)
        @test counts[1][] == 50
        @test counts[2][] == 50
    end

    @testset "Error without callback type" begin
        @test_throws MethodError mcmc_callback()
    end

    @testset "Adding callbacks with push!!" begin
        counts = [Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1

        cb = mcmc_callback(cb1)
        cb = BangBang.push!!(cb, cb2)

        for _ in 1:5
            cb(nothing, nothing, nothing, nothing, nothing, 1)
        end
        @test counts[1][] == 5
        @test counts[2][] == 5
    end

    @testset "Callable struct callback" begin
        struct CountingCallback
            count::Ref{Int}
        end
        function (cb::CountingCallback)(args...; kwargs...)
            return cb.count[] += 1
        end

        counter = CountingCallback(Ref(0))
        cb = mcmc_callback(counter)

        @test cb isa AbstractMCMC.MultiCallback
        chain = sample(MyModel(), MySampler(), 25; callback=cb)
        @test counter.count[] == 25
    end
end

########################
### Defaults Merging ###
########################

@testset "Defaults merging" begin
    @testset "merge_with_defaults" begin
        defaults = (; a=1, b=2, c=3)

        # Partial override
        result = AbstractMCMC.merge_with_defaults((; b=10), defaults)
        @test result == (; a=1, b=10, c=3)

        # Full override
        result = AbstractMCMC.merge_with_defaults((; a=10, b=20, c=30), defaults)
        @test result == (; a=10, b=20, c=30)

        # No override (nothing)
        result = AbstractMCMC.merge_with_defaults(nothing, defaults)
        @test result == defaults
    end

    @testset "DEFAULT_STATS_OPTIONS" begin
        @test AbstractMCMC.DEFAULT_STATS_OPTIONS.thin == 0
        @test AbstractMCMC.DEFAULT_STATS_OPTIONS.skip == 0
        @test AbstractMCMC.DEFAULT_STATS_OPTIONS.window == typemax(Int)
    end

    @testset "DEFAULT_NAME_FILTER" begin
        @test AbstractMCMC.DEFAULT_NAME_FILTER.include == String[]
        @test AbstractMCMC.DEFAULT_NAME_FILTER.exclude == String[]
        @test AbstractMCMC.DEFAULT_NAME_FILTER.stats == false
        @test AbstractMCMC.DEFAULT_NAME_FILTER.extras == false
    end
end

######################
### Internal Types ###
######################

@testset "MultiCallback" begin
    counts = [Ref(0), Ref(0)]
    cb1 = (args...; kwargs...) -> counts[1][] += 1
    cb2 = (args...; kwargs...) -> counts[2][] += 1

    multi = AbstractMCMC.MultiCallback(cb1, cb2)
    multi(nothing, nothing, nothing, nothing, nothing, 1)

    @test counts[1][] == 1
    @test counts[2][] == 1
end

@testset "NameFilter" begin
    @testset "Include only" begin
        f = AbstractMCMC.NameFilter(; include=["a", "b"])
        @test f("a") == true
        @test f("b") == true
        @test f("c") == false
    end

    @testset "Exclude only" begin
        f = AbstractMCMC.NameFilter(; exclude=["x", "y"])
        @test f("a") == true
        @test f("x") == false
        @test f("y") == false
    end

    @testset "Include and exclude (non-overlapping)" begin
        f = AbstractMCMC.NameFilter(; include=["a", "b"], exclude=["x", "y"])
        @test f("a") == true
        @test f("b") == true
        @test f("x") == false
        @test f("y") == false
        @test f("c") == false
    end

    @testset "Include and exclude (overlapping errors)" begin
        @test_throws ErrorException AbstractMCMC.NameFilter(;
            include=["a", "b", "c"], exclude=["c"]
        )
    end

    @testset "No filter" begin
        f = AbstractMCMC.NameFilter()
        @test f("anything") == true
    end

    @testset "Two argument form" begin
        f = AbstractMCMC.NameFilter(; include=["a", "b"])
        @test f("a", 1.0) == true
        @test f("c", 2.0) == false
    end

    @testset "Symbol names (from NamedTuple iteration)" begin
        f = AbstractMCMC.NameFilter(; include=["a", "b"])
        @test f(:a) == true
        @test f(:c) == false
    end
end

#########################
### ParamsWithStats   ###
#########################

@testset "ParamsWithStats" begin
    @testset "Constructor from NamedTuple" begin
        pws = AbstractMCMC.ParamsWithStats(
            (a=1.0, b=2.0), (lp=-10.0,), NamedTuple()
        )
        @test pws isa AbstractMCMC.ParamsWithStats
        @test pws.params == (a=1.0, b=2.0)
        @test pws.stats == (lp=-10.0,)
        @test pws.extras == NamedTuple()
    end

    @testset "Constructor from Vector{Real} - default names" begin
        pws = AbstractMCMC.ParamsWithStats(
            [1.0, 2.0, 3.0], NamedTuple(), NamedTuple()
        )
        @test pws.params == (var"θ[1]"=1.0, var"θ[2]"=2.0, var"θ[3]"=3.0)
    end

    @testset "Constructor from Vector{Pair} - named" begin
        pws = AbstractMCMC.ParamsWithStats(
            ["μ" => 1.0, "σ" => 2.0], NamedTuple(), NamedTuple()
        )
        @test pws.params == (μ=1.0, σ=2.0)
    end

    @testset "Constructor from state" begin
        state = 5
        pws = AbstractMCMC.ParamsWithStats(
            MyModel(), MySampler(), nothing, state; params=true, stats=true
        )
        @test pws isa AbstractMCMC.ParamsWithStats
        @test pws.params == NamedTuple()  # Empty vector becomes empty NamedTuple
        @test pws.stats == (iteration=5,)
        @test pws.extras == NamedTuple()
    end

    @testset "Copy constructor with selection" begin
        pws = AbstractMCMC.ParamsWithStats(
            (a=1.0,), (lp=-10.0,), NamedTuple()
        )

        # Select only params
        pws_params = AbstractMCMC.ParamsWithStats(pws; params=true, stats=false)
        @test pws_params.params == (a=1.0,)
        @test pws_params.stats == NamedTuple()

        # Select only stats
        pws_stats = AbstractMCMC.ParamsWithStats(pws; params=false, stats=true)
        @test pws_stats.params == NamedTuple()
        @test pws_stats.stats == (lp=-10.0,)
    end

    @testset "Base.pairs iteration" begin
        pws = AbstractMCMC.ParamsWithStats(
            (a=1.0, b=2.0), (lp=-10.0,), NamedTuple()
        )
        pairs_list = collect(Base.pairs(pws))
        @test length(pairs_list) == 3
        @test (:a => 1.0) in pairs_list
        @test (:b => 2.0) in pairs_list
        @test (:lp => -10.0) in pairs_list
    end

    @testset "Base.isempty" begin
        pws_full = AbstractMCMC.ParamsWithStats(
            (a=1.0,), (lp=-10.0,), NamedTuple()
        )
        @test !isempty(pws_full)

        pws_empty = AbstractMCMC.ParamsWithStats(
            NamedTuple(), NamedTuple(), NamedTuple()
        )
        @test isempty(pws_empty)
    end

    @testset "Illegal states are unrepresentable" begin
        # Should not be able to construct with arbitrary types
        @test_throws MethodError AbstractMCMC.ParamsWithStats(1, 2, 3)
        @test_throws MethodError AbstractMCMC.ParamsWithStats("bad", NamedTuple(), NamedTuple())
    end
end

using OnlineStats

#############################
### TensorBoard Extension ###
#############################

using TensorBoardLogger

@testset "TensorBoard Extension" begin
    @testset "mcmc_callback with explicit TBLogger" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(; logger=logger)
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback requires logger argument" begin
        @test_throws UndefKeywordError mcmc_callback()
    end

    @testset "mcmc_callback with stats=true" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(; logger=logger, stats=true)
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback with stats=:default" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(; logger=logger, stats=:default)
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback with explicit OnlineStats" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(; logger=logger, stats=(Mean(), Variance()))
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback with stats_options" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)

        # Test partial stats_options (merges with defaults)
        cb = mcmc_callback(; logger=logger, stats=true, stats_options=(thin=5,))
        @test cb isa AbstractMCMC.MultiCallback

        # Test full stats_options
        cb = mcmc_callback(;
            logger=logger, stats=true, stats_options=(thin=5, skip=100, window=1000)
        )
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback with name_filter" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)

        # Test partial name_filter
        cb = mcmc_callback(; logger=logger, name_filter=(include=["mu", "sigma"],))
        @test cb isa AbstractMCMC.MultiCallback

        # Test full name_filter
        cb = mcmc_callback(;
            logger=logger,
            name_filter=(
                include=["mu", "sigma"], exclude=["internal"], stats=true, extras=true
            ),
        )
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "mcmc_callback with all options" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(;
            logger=logger,
            stats=true,
            stats_options=(skip=100, thin=5),
            name_filter=(exclude=["_internal"], extras=true),
        )
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "TensorBoard callback with sample" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        cb = mcmc_callback(; logger=logger)

        # Should complete without error
        chain = sample(MyModel(), MySampler(), 20; callback=cb)
        @test length(chain) == 20
    end

    @testset "mcmc_callback with custom AbstractLogger" begin
        # Use TBLogger as our custom logger (could be any AbstractLogger)
        logdir = mktempdir()
        custom_logger = TensorBoardLogger.TBLogger(logdir; min_level=Logging.Info)

        cb = mcmc_callback(; logger=custom_logger)
        @test cb isa AbstractMCMC.MultiCallback

        # Should work with sampling
        chain = sample(MyModel(), MySampler(), 20; callback=cb)
        @test length(chain) == 20
    end

    @testset "Stats with stats=true works when OnlineStats loaded" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        # OnlineStats is already loaded, so this should work
        cb = mcmc_callback(; logger=logger, stats=true)
        @test cb isa AbstractMCMC.MultiCallback

        # Also works with explicit OnlineStat
        cb = mcmc_callback(; logger=logger, stats=Mean())
        @test cb isa AbstractMCMC.MultiCallback
    end

    @testset "ParamsWithStats default implementation" begin
        struct MockState
            params::Vector{Float64}
        end
        AbstractMCMC.getparams(s::MockState) = s.params

        state = MockState([10.0, 20.0])
        pws = AbstractMCMC.ParamsWithStats(nothing, nothing, nothing, state; params=true)
        result = collect(Base.pairs(pws))

        @test length(result) == 2
        @test result[1] == (Symbol("θ[1]") => 10.0)
        @test result[2] == (Symbol("θ[2]") => 20.0)
    end
end

#########################
### Integration Tests ###
#########################

@testset "Integration" begin
    @testset "Callback receives correct iteration" begin
        iterations = Int[]
        cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
            push!(iterations, iteration)
        end

        chain = sample(MyModel(), MySampler(), 50; callback=cb)
        @test iterations == 1:50
    end

    @testset "Multiple callbacks all execute" begin
        results = Dict{Symbol,Int}(:cb1 => 0, :cb2 => 0, :cb3 => 0)

        cb = mcmc_callback(
            (args...; kwargs...) -> results[:cb1] += 1,
            (args...; kwargs...) -> results[:cb2] += 1,
            (args...; kwargs...) -> results[:cb3] += 1,
        )

        chain = sample(MyModel(), MySampler(), 30; callback=cb)
        @test results[:cb1] == 30
        @test results[:cb2] == 30
        @test results[:cb3] == 30
    end

    @testset "Combining TensorBoard with custom callback" begin
        logdir = mktempdir()
        logger = TBLogger(logdir)
        count = Ref(0)
        custom = (args...; kwargs...) -> count[] += 1

        tb_cb = mcmc_callback(; logger=logger)
        combined = mcmc_callback(tb_cb, custom)

        @test length(combined.callbacks) == 2
    end
end
