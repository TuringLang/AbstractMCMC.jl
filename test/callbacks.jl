# Tests for the mcmc_callback unified API
# These tests cover: stats, stats_options, name_filter, and basic functionality

####################################
### Basic Callback Functionality ###
####################################

@testset "Basic mcmc_callback" begin
    @testset "Function callback" begin
        count = Ref(0)
        cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
            count[] += 1
        end

        @test cb isa AbstractMCMC.Callback
        chain = sample(MyModel(), MySampler(), 100; callback=cb)
        @test count[] == 100
    end

    @testset "Multiple function callbacks" begin
        counts = [Ref(0), Ref(0)]
        cb1 = (args...; kwargs...) -> counts[1][] += 1
        cb2 = (args...; kwargs...) -> counts[2][] += 1

        cb = mcmc_callback(cb1, cb2)
        @test cb isa AbstractMCMC.Callback

        chain = sample(MyModel(), MySampler(), 50; callback=cb)
        @test counts[1][] == 50
        @test counts[2][] == 50
    end

    @testset "Error without callback type" begin
        @test_throws ArgumentError mcmc_callback()
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
        @test AbstractMCMC.DEFAULT_NAME_FILTER.extras == false
        @test AbstractMCMC.DEFAULT_NAME_FILTER.hyperparams == false
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

    @testset "Include and exclude" begin
        f = AbstractMCMC.NameFilter(; include=["a", "b", "c"], exclude=["c"])
        @test f("a") == true
        @test f("c") == false
        @test f("d") == false
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
end

@testset "default_param_names_for_values" begin
    names = collect(AbstractMCMC.default_param_names_for_values([1.0, 2.0, 3.0]))
    @test names == ["θ[1]", "θ[2]", "θ[3]"]
end

using OnlineStats

#############################
### TensorBoard Extension ###
#############################

using TensorBoardLogger

@testset "TensorBoard Extension" begin
    @testset "mcmc_callback with logger=:TBLogger" begin
        logdir = mktempdir()
        cb = mcmc_callback(; logger=:TBLogger, logdir=logdir)
        @test cb isa AbstractMCMC.Callback
    end

    @testset "mcmc_callback with logdir only (infers TBLogger)" begin
        logdir = mktempdir()
        cb = mcmc_callback(; logdir=logdir)
        @test cb isa AbstractMCMC.Callback
    end

    @testset "mcmc_callback with stats" begin
        logdir = mktempdir()
        cb = mcmc_callback(; logdir=logdir, stats=(Mean(), Variance()))
        @test cb isa AbstractMCMC.Callback
    end

    @testset "mcmc_callback with stats_options" begin
        logdir = mktempdir()

        # Test partial stats_options (merges with defaults)
        cb = mcmc_callback(; logdir=logdir, stats_options=(thin=5,))
        @test cb isa AbstractMCMC.Callback

        # Test full stats_options
        cb = mcmc_callback(; logdir=logdir, stats_options=(thin=5, skip=100, window=1000))
        @test cb isa AbstractMCMC.Callback
    end

    @testset "mcmc_callback with name_filter" begin
        logdir = mktempdir()

        # Test partial name_filter
        cb = mcmc_callback(; logdir=logdir, name_filter=(include=["mu", "sigma"],))
        @test cb isa AbstractMCMC.Callback

        # Test full name_filter
        cb = mcmc_callback(;
            logdir=logdir,
            name_filter=(
                include=["mu", "sigma"], exclude=["internal"], extras=true, hyperparams=true
            ),
        )
        @test cb isa AbstractMCMC.Callback
    end

    @testset "mcmc_callback with all options" begin
        logdir = mktempdir()
        cb = mcmc_callback(;
            logdir=logdir,
            stats=(Mean(), Variance(), KHist(50)),
            stats_options=(skip=100, thin=5),
            name_filter=(exclude=["_internal"], extras=true),
        )
        @test cb isa AbstractMCMC.Callback
    end

    @testset "TensorBoard callback with sample" begin
        logdir = mktempdir()
        cb = mcmc_callback(; logdir=logdir)

        # Should complete without error
        chain = sample(MyModel(), MySampler(), 20; callback=cb)
        @test length(chain) == 20
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
        count = Ref(0)
        custom = (args...; kwargs...) -> count[] += 1

        tb_cb = mcmc_callback(; logdir=logdir)
        combined = mcmc_callback(tb_cb, custom)

        @test length(combined.multi.callbacks) == 2
    end
end
