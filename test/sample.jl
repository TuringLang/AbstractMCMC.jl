@testset "sample.jl" begin
    @testset "Basic sampling" begin
        @testset "REPL" begin
            empty!(LOGGERS)

            Random.seed!(1234)
            N = 1_000
            chain = sample(MyModel(), MySampler(), N; loggers=true)

            @test length(LOGGERS) == 1
            logger = first(LOGGERS)
            @test logger isa TeeLogger
            @test logger.loggers[1].logger isa
                (Sys.iswindows() && VERSION < v"1.5.3" ? ProgressLogger : TerminalLogger)
            @test logger.loggers[2].logger === CURRENT_LOGGER
            @test Logging.current_logger() === CURRENT_LOGGER

            # test output type and size
            @test chain isa Vector{<:MySample}
            @test length(chain) == N

            # test some statistical properties
            tail_chain = @view chain[2:end]
            @test mean(x.a for x in tail_chain) ≈ 0.5 atol = 6e-2
            @test var(x.a for x in tail_chain) ≈ 1 / 12 atol = 5e-3
            @test mean(x.b for x in tail_chain) ≈ 0.0 atol = 5e-2
            @test var(x.b for x in tail_chain) ≈ 1 atol = 6e-2

            # initial parameters
            chain = sample(
                MyModel(), MySampler(), 3; progress=false, init_params=(b=3.2, a=-1.8)
            )
            @test chain[1].a == -1.8
            @test chain[1].b == 3.2
        end

        @testset "Juno" begin
            empty!(LOGGERS)

            Random.seed!(1234)
            N = 10

            logger = JunoProgressLogger()
            Logging.with_logger(logger) do
                sample(MyModel(), MySampler(), N; loggers=true)
            end

            @test length(LOGGERS) == 1
            @test first(LOGGERS) === logger
            @test Logging.current_logger() === CURRENT_LOGGER
        end

        @testset "IJulia" begin
            # emulate running IJulia kernel
            @eval IJulia begin
                inited = true
            end

            empty!(LOGGERS)

            Random.seed!(1234)
            N = 10
            sample(MyModel(), MySampler(), N; loggers=true)

            @test length(LOGGERS) == 1
            logger = first(LOGGERS)
            @test logger isa TeeLogger
            @test logger.loggers[1].logger isa ProgressLogger
            @test logger.loggers[2].logger === CURRENT_LOGGER
            @test Logging.current_logger() === CURRENT_LOGGER

            @eval IJulia begin
                inited = false
            end
        end

        @testset "Custom logger" begin
            empty!(LOGGERS)

            Random.seed!(1234)
            N = 10

            logger = Logging.ConsoleLogger(stderr, Logging.LogLevel(-1))
            Logging.with_logger(logger) do
                sample(MyModel(), MySampler(), N; loggers=true)
            end

            @test length(LOGGERS) == 1
            @test first(LOGGERS) === logger
            @test Logging.current_logger() === CURRENT_LOGGER
        end

        @testset "Suppress output" begin
            logs, _ = collect_test_logs(; min_level=Logging.LogLevel(-1)) do
                sample(MyModel(), MySampler(), 100; progress=false)
            end
            @test all(l.level > Logging.LogLevel(-1) for l in logs)

            # disable progress logging globally
            @test !(@test_logs (:info, "progress logging is disabled globally") AbstractMCMC.setprogress!(
                false
            ))
            @test !AbstractMCMC.PROGRESS[]

            logs, _ = collect_test_logs(; min_level=Logging.LogLevel(-1)) do
                sample(MyModel(), MySampler(), 100)
            end
            @test all(l.level > Logging.LogLevel(-1) for l in logs)

            # enable progress logging globally
            @test (@test_logs (:info, "progress logging is enabled globally") AbstractMCMC.setprogress!(
                true
            ))
            @test AbstractMCMC.PROGRESS[]
        end
    end

    @testset "Multithreaded sampling" begin
        if Threads.nthreads() == 1
            warnregex = r"^Only a single thread available"
            @test_logs (:warn, warnregex) sample(
                MyModel(), MySampler(), MCMCThreads(), 10, 10
            )
        end

        # No dedicated chains type
        N = 10_000
        chains = sample(MyModel(), MySampler(), MCMCThreads(), N, 1000)
        @test chains isa Vector{<:Vector{<:MySample}}
        @test length(chains) == 1000
        @test all(length(x) == N for x in chains)

        Random.seed!(1234)
        chains = sample(MyModel(), MySampler(), MCMCThreads(), N, 1000; chain_type=MyChain)

        # test output type and size
        @test chains isa Vector{<:MyChain}
        @test length(chains) == 1000
        @test all(x -> length(x.as) == length(x.bs) == N, chains)
        @test all(ismissing(x.as[1]) for x in chains)

        # test some statistical properties
        @test all(x -> isapprox(mean(@view x.as[2:end]), 0.5; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.as[2:end]), 1 / 12; atol=5e-3), chains)
        @test all(x -> isapprox(mean(@view x.bs[2:end]), 0; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.bs[2:end]), 1; atol=1e-1), chains)

        # test reproducibility
        Random.seed!(1234)
        chains2 = sample(MyModel(), MySampler(), MCMCThreads(), N, 1000; chain_type=MyChain)
        @test all(ismissing(x.as[1]) for x in chains2)
        @test all(c1.as[i] == c2.as[i] for (c1, c2) in zip(chains, chains2), i in 2:N)
        @test all(c1.bs[i] == c2.bs[i] for (c1, c2) in zip(chains, chains2), i in 1:N)

        # Unexpected order of arguments.
        str = "Number of chains (10) is greater than number of samples per chain (5)"
        @test_logs (:warn, str) match_mode = :any sample(
            MyModel(), MySampler(), MCMCThreads(), 5, 10; chain_type=MyChain
        )

        # Suppress output.
        logs, _ = collect_test_logs(; min_level=Logging.LogLevel(-1)) do
            sample(
                MyModel(),
                MySampler(),
                MCMCThreads(),
                10_000,
                1000;
                progress=false,
                chain_type=MyChain,
            )
        end
        @test all(l.level > Logging.LogLevel(-1) for l in logs)

        # Smoke test for nchains < nthreads
        if Threads.nthreads() == 2
            sample(MyModel(), MySampler(), MCMCThreads(), N, 1)
        end

        # initial parameters
        init_params = [(b=randn(), a=rand()) for _ in 1:100]
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCThreads(),
            3,
            100;
            progress=false,
            init_params=init_params,
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == params.a && chain[1].b == params.b for
            (chain, params) in zip(chains, init_params)
        )

        init_params = (a=randn(), b=rand())
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCThreads(),
            3,
            100;
            progress=false,
            init_params=Iterators.repeated(init_params),
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == init_params.a && chain[1].b == init_params.b for chain in chains
        )
    end

    @testset "Multicore sampling" begin
        if nworkers() == 1
            warnregex = r"^Only a single process available"
            @test_logs (:warn, warnregex) sample(
                MyModel(), MySampler(), MCMCDistributed(), 10, 10; chain_type=MyChain
            )
        end

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

        # No dedicated chains type
        N = 10_000
        chains = sample(MyModel(), MySampler(), MCMCThreads(), N, 1000)
        @test chains isa Vector{<:Vector{<:MySample}}
        @test length(chains) == 1000
        @test all(length(x) == N for x in chains)

        Random.seed!(1234)
        chains = sample(
            MyModel(), MySampler(), MCMCDistributed(), N, 1000; chain_type=MyChain
        )

        # Test output type and size.
        @test chains isa Vector{<:MyChain}
        @test all(ismissing(c.as[1]) for c in chains)
        @test length(chains) == 1000
        @test all(x -> length(x.as) == length(x.bs) == N, chains)

        # Test some statistical properties.
        @test all(x -> isapprox(mean(@view x.as[2:end]), 0.5; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.as[2:end]), 1 / 12; atol=5e-3), chains)
        @test all(x -> isapprox(mean(@view x.bs[2:end]), 0; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.bs[2:end]), 1; atol=1e-1), chains)

        # Test reproducibility.
        Random.seed!(1234)
        chains2 = sample(
            MyModel(), MySampler(), MCMCDistributed(), N, 1000; chain_type=MyChain
        )
        @test all(ismissing(c.as[1]) for c in chains2)
        @test all(c1.as[i] == c2.as[i] for (c1, c2) in zip(chains, chains2), i in 2:N)
        @test all(c1.bs[i] == c2.bs[i] for (c1, c2) in zip(chains, chains2), i in 1:N)

        # Unexpected order of arguments.
        str = "Number of chains (10) is greater than number of samples per chain (5)"
        @test_logs (:warn, str) match_mode = :any sample(
            MyModel(), MySampler(), MCMCDistributed(), 5, 10; chain_type=MyChain
        )

        # Suppress output.
        logs, _ = collect_test_logs(; min_level=Logging.LogLevel(-1)) do
            sample(
                MyModel(),
                MySampler(),
                MCMCDistributed(),
                10_000,
                100;
                progress=false,
                chain_type=MyChain,
            )
        end
        @test all(l.level > Logging.LogLevel(-1) for l in logs)

        # initial parameters
        init_params = [(a=randn(), b=rand()) for _ in 1:100]
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCDistributed(),
            3,
            100;
            progress=false,
            init_params=init_params,
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == params.a && chain[1].b == params.b for
            (chain, params) in zip(chains, init_params)
        )

        init_params = (b=randn(), a=rand())
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCDistributed(),
            3,
            100;
            progress=false,
            init_params=Iterators.repeated(init_params),
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == init_params.a && chain[1].b == init_params.b for chain in chains
        )

        # Remove workers
        rmprocs(pids...)
    end

    @testset "Serial sampling" begin
        # No dedicated chains type
        N = 10_000
        chains = sample(MyModel(), MySampler(), MCMCSerial(), N, 1000)
        @test chains isa Vector{<:Vector{<:MySample}}
        @test length(chains) == 1000
        @test all(length(x) == N for x in chains)

        Random.seed!(1234)
        chains = sample(MyModel(), MySampler(), MCMCSerial(), N, 1000; chain_type=MyChain)

        # Test output type and size.
        @test chains isa Vector{<:MyChain}
        @test all(ismissing(c.as[1]) for c in chains)
        @test length(chains) == 1000
        @test all(x -> length(x.as) == length(x.bs) == N, chains)

        # Test some statistical properties.
        @test all(x -> isapprox(mean(@view x.as[2:end]), 0.5; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.as[2:end]), 1 / 12; atol=5e-3), chains)
        @test all(x -> isapprox(mean(@view x.bs[2:end]), 0; atol=5e-2), chains)
        @test all(x -> isapprox(var(@view x.bs[2:end]), 1; atol=1e-1), chains)

        # Test reproducibility.
        Random.seed!(1234)
        chains2 = sample(MyModel(), MySampler(), MCMCSerial(), N, 1000; chain_type=MyChain)
        @test all(ismissing(c.as[1]) for c in chains2)
        @test all(c1.as[i] == c2.as[i] for (c1, c2) in zip(chains, chains2), i in 2:N)
        @test all(c1.bs[i] == c2.bs[i] for (c1, c2) in zip(chains, chains2), i in 1:N)

        # Unexpected order of arguments.
        str = "Number of chains (10) is greater than number of samples per chain (5)"
        @test_logs (:warn, str) match_mode = :any sample(
            MyModel(), MySampler(), MCMCSerial(), 5, 10; chain_type=MyChain
        )

        # Suppress output.
        logs, _ = collect_test_logs(; min_level=Logging.LogLevel(-1)) do
            sample(
                MyModel(),
                MySampler(),
                MCMCSerial(),
                10_000,
                100;
                progress=false,
                chain_type=MyChain,
            )
        end
        @test all(l.level > Logging.LogLevel(-1) for l in logs)

        # initial parameters
        init_params = [(a=rand(), b=randn()) for _ in 1:100]
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCSerial(),
            3,
            100;
            progress=false,
            init_params=init_params,
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == params.a && chain[1].b == params.b for
            (chain, params) in zip(chains, init_params)
        )

        init_params = (b=rand(), a=randn())
        chains = sample(
            MyModel(),
            MySampler(),
            MCMCSerial(),
            3,
            100;
            progress=false,
            init_params=Iterators.repeated(init_params),
        )
        @test length(chains) == 100
        @test all(
            chain[1].a == init_params.a && chain[1].b == init_params.b for chain in chains
        )
    end

    @testset "Ensemble sampling: Reproducibility" begin
        N = 1_000
        nchains = 10

        # Serial sampling
        Random.seed!(1234)
        chains_serial = sample(
            MyModel(),
            MySampler(),
            MCMCSerial(),
            N,
            nchains;
            progress=false,
            chain_type=MyChain,
        )
        @test all(ismissing(c.as[1]) for c in chains_serial)

        # Multi-threaded sampling
        Random.seed!(1234)
        chains_threads = sample(
            MyModel(),
            MySampler(),
            MCMCThreads(),
            N,
            nchains;
            progress=false,
            chain_type=MyChain,
        )
        @test all(ismissing(c.as[1]) for c in chains_threads)
        @test all(
            c1.as[i] == c2.as[i] for (c1, c2) in zip(chains_serial, chains_threads),
            i in 2:N
        )
        @test all(
            c1.bs[i] == c2.bs[i] for (c1, c2) in zip(chains_serial, chains_threads),
            i in 1:N
        )

        # Multi-core sampling
        Random.seed!(1234)
        chains_distributed = sample(
            MyModel(),
            MySampler(),
            MCMCDistributed(),
            N,
            nchains;
            progress=false,
            chain_type=MyChain,
        )
        @test all(ismissing(c.as[1]) for c in chains_distributed)
        @test all(
            c1.as[i] == c2.as[i] for (c1, c2) in zip(chains_serial, chains_distributed),
            i in 2:N
        )
        @test all(
            c1.bs[i] == c2.bs[i] for (c1, c2) in zip(chains_serial, chains_distributed),
            i in 1:N
        )
    end

    @testset "Chain constructors" begin
        chain1 = sample(MyModel(), MySampler(), 100)
        chain2 = sample(MyModel(), MySampler(), 100; chain_type=MyChain)

        @test chain1 isa Vector{<:MySample}
        @test chain2 isa MyChain
    end

    @testset "Sample stats" begin
        chain = sample(MyModel(), MySampler(), 1000; chain_type=MyChain)

        @test chain.stats.stop >= chain.stats.start
        @test chain.stats.duration == chain.stats.stop - chain.stats.start
    end

    @testset "Discard initial samples" begin
        # Create a chain and discard initial samples.
        Random.seed!(1234)
        N = 100
        discard_initial = 50
        chain = sample(MyModel(), MySampler(), N; discard_initial=discard_initial)
        @test length(chain) == N
        @test !ismissing(chain[1].a)

        # Repeat sampling without discarding initial samples.
        # On Julia < 1.6 progress logging changes the global RNG and hence is enabled here.
        # https://github.com/TuringLang/AbstractMCMC.jl/pull/102#issuecomment-1142253258
        Random.seed!(1234)
        ref_chain = sample(
            MyModel(), MySampler(), N + discard_initial; progress=VERSION < v"1.6"
        )
        @test all(chain[i].a == ref_chain[i + discard_initial].a for i in 1:N)
        @test all(chain[i].b == ref_chain[i + discard_initial].b for i in 1:N)
    end

    @testset "Thin chain by a factor of `thinning`" begin
        # Run a thinned chain with `N` samples thinned by factor of `thinning`.
        Random.seed!(100)
        N = 100
        thinning = 3
        chain = sample(MyModel(), MySampler(), N; thinning=thinning)
        @test length(chain) == N
        @test ismissing(chain[1].a)

        # Repeat sampling without thinning.
        # On Julia < 1.6 progress logging changes the global RNG and hence is enabled here.
        # https://github.com/TuringLang/AbstractMCMC.jl/pull/102#issuecomment-1142253258
        Random.seed!(100)
        ref_chain = sample(MyModel(), MySampler(), N * thinning; progress=VERSION < v"1.6")
        @test all(chain[i].a == ref_chain[(i - 1) * thinning + 1].a for i in 2:N)
        @test all(chain[i].b == ref_chain[(i - 1) * thinning + 1].b for i in 1:N)
    end

    @testset "Sample without predetermined N" begin
        Random.seed!(1234)
        chain = sample(MyModel(), MySampler())
        bmean = mean(x.b for x in chain)
        @test ismissing(chain[1].a)
        @test abs(bmean) <= 0.001 || length(chain) == 10_000

        # Discard initial samples.
        Random.seed!(1234)
        discard_initial = 50
        chain = sample(MyModel(), MySampler(); discard_initial=discard_initial)
        bmean = mean(x.b for x in chain)
        @test !ismissing(chain[1].a)
        @test abs(bmean) <= 0.001 || length(chain) == 10_000

        # On Julia < 1.6 progress logging changes the global RNG and hence is enabled here.
        # https://github.com/TuringLang/AbstractMCMC.jl/pull/102#issuecomment-1142253258
        Random.seed!(1234)
        N = length(chain)
        ref_chain = sample(
            MyModel(),
            MySampler(),
            N;
            discard_initial=discard_initial,
            progress=VERSION < v"1.6",
        )
        @test all(chain[i].a == ref_chain[i].a for i in 1:N)
        @test all(chain[i].b == ref_chain[i].b for i in 1:N)

        # Thin chain by a factor of `thinning`.
        Random.seed!(1234)
        thinning = 3
        chain = sample(MyModel(), MySampler(); thinning=thinning)
        bmean = mean(x.b for x in chain)
        @test ismissing(chain[1].a)
        @test abs(bmean) <= 0.001 || length(chain) == 10_000

        # On Julia < 1.6 progress logging changes the global RNG and hence is enabled here.
        # https://github.com/TuringLang/AbstractMCMC.jl/pull/102#issuecomment-1142253258
        Random.seed!(1234)
        N = length(chain)
        ref_chain = sample(
            MyModel(), MySampler(), N; thinning=thinning, progress=VERSION < v"1.6"
        )
        @test all(chain[i].a == ref_chain[i].a for i in 2:N)
        @test all(chain[i].b == ref_chain[i].b for i in 1:N)
    end

    @testset "Sample vector of `NamedTuple`s" begin
        chain = sample(MyModel(), MySampler(), 1_000; chain_type=Vector{NamedTuple})
        # Check output type
        @test chain isa Vector{<:NamedTuple}
        @test length(chain) == 1_000
        @test all(keys(x) == (:a, :b) for x in chain)

        # Check some statistical properties
        @test ismissing(chain[1].a)
        @test mean(x.a for x in view(chain, 2:1_000)) ≈ 0.5 atol = 6e-2
        @test var(x.a for x in view(chain, 2:1_000)) ≈ 1 / 12 atol = 1e-2
        @test mean(x.b for x in chain) ≈ 0 atol = 0.1
        @test var(x.b for x in chain) ≈ 1 atol = 0.15
    end

    @testset "Testing callbacks" begin
        function count_iterations(
            rng, model, sampler, sample, state, i; iter_array, kwargs...
        )
            return push!(iter_array, i)
        end
        N = 100
        it_array = Float64[]
        sample(MyModel(), MySampler(), N; callback=count_iterations, iter_array=it_array)
        @test it_array == collect(1:N)

        # sampling without predetermined N
        it_array = Float64[]
        chain = sample(
            MyModel(), MySampler(); callback=count_iterations, iter_array=it_array
        )
        @test it_array == collect(1:size(chain, 1))
    end
end
