"""
    LogDensityModel <: AbstractMCMC.AbstractModel

Wrapper around something that implements the `LogDensityProblem` interface.

This itself then implements the `LogDensityProblem` interface by simply deferring to the wrapped object.
"""
struct LogDensityModel{L} <: AbstractModel
    logdensity::L
end

function LogDensityProblems.dimension(model::LogDensityModel)
    return LogDensityProblems.dimension(model.logdensity)
end
function LogDensityProblems.capabilities(model::LogDensityModel)
    return LogDensityProblems.capabilities(model.logdensity)
end
function LogDensityProblems.logdensity(model::LogDensityModel, x)
    return LogDensityProblems.logdensity(model.logdensity, x)
end
function LogDensityProblems.logdensity_and_gradient(model::LogDensityModel, x)
    return LogDensityProblems.logdensity_and_gradient(model.logdensity, x)
end
