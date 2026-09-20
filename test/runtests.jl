using AbstractMCMC
using BangBang
using ConsoleProgressMonitor: ProgressLogger
using LogDensityProblems
using LoggingExtras: TeeLogger, EarlyFilteredLogger
using TerminalLoggers: TerminalLogger
using FillArrays: FillArrays

using Distributed
using Logging: Logging
using Random
using Statistics
using Test
using Test: collect_test_logs

const LOGGERS = Set()
const CURRENT_LOGGER = Logging.current_logger()

# Logger selection only reads `Main.IJulia.inited`; no notebook runtime is needed.
module IJulia
inited = false
end

include("utils.jl")

@testset "AbstractMCMC" begin
    include("sample.jl")
    include("stepper.jl")
    include("logdensityproblems.jl")
end

@testset "Callbacks" begin
    include("callbacks.jl")
end
