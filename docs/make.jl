using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using AbstractMCMC
using Random

DocMeta.setdocmeta!(
    AbstractMCMC,
    :DocTestSetup,
    :(using AbstractMCMC);
    recursive=true,
)

makedocs(;
    sitename="AbstractMCMC",
    format=Documenter.HTML(),
    modules=[AbstractMCMC],
    pages=[
        "Home" => "index.md",
        "api.md",
        "design.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/TuringLang/AbstractMCMC.jl.git", push_preview=true
)
