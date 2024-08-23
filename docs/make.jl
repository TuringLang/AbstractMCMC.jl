using AbstractMCMC
using Documenter
using Random

DocMeta.setdocmeta!(AbstractMCMC, :DocTestSetup, :(using AbstractMCMC); recursive=true)

makedocs(;
    sitename="AbstractMCMC",
    format=Documenter.HTML(),
    modules=[AbstractMCMC],
    pages=["Home" => "index.md", "api.md", "design.md", "gibbs.md"],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/TuringLang/AbstractMCMC.jl.git", push_preview=true)
