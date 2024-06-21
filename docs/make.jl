using AbstractMCMC
using Documenter
using Random

DocMeta.setdocmeta!(AbstractMCMC, :DocTestSetup, :(using AbstractMCMC); recursive=true)

makedocs(;
    sitename="AbstractMCMC",
    format=Documenter.HTML(),
    modules=[AbstractMCMC],
    pages=["Home" => "index.md", "api.md", "design.md"],
    checkdocs=:exports,
)

# Insert navbar in each html file
run(
    `sh -c "curl -s https://raw.githubusercontent.com/TuringLang/turinglang.github.io/main/assets/scripts/insert_navbar.sh | bash -s docs/build"`,
)

deploydocs(; repo="github.com/TuringLang/AbstractMCMC.jl.git", push_preview=true)
