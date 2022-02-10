using AbstractMCMC
using Atom.Progress: JunoProgressLogger
using ConsoleProgressMonitor: ProgressLogger
using IJulia
using LoggingExtras: TeeLogger, EarlyFilteredLogger
using TerminalLoggers: TerminalLogger
using Transducers

using Distributed
import Logging
using Random
using Statistics
using Test
using Test: collect_test_logs

const LOGGERS = Set()
const CURRENT_LOGGER = Logging.current_logger()

include("utils.jl")

@testset "AbstractMCMC" begin
    include("sample.jl")
    include("stepper.jl")
    include("transducer.jl")
end
