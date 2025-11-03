module AbstractMCMCChainsTests

using AbstractMCMC: AbstractMCMC
using Test

# This is a test mock: it minimally satisfies the AbstractChains interface. We use this to
# test our `test_interface` function, i.e., to ensure that something that satisfies the
# interface passes the test.
# See: https://invenia.github.io/blog/2020/11/06/interfacetesting/
struct MockChain <: AbstractMCMC.AbstractChains
    iter_indices::Vector{Int}
    chain_indices::Vector{Int}
    data::Dict{Symbol,Matrix{Float64}}
end
const MOCK = MockChain(1:10, 1:3, Dict(:param1 => rand(10, 3), :param2 => rand(10, 3)))
AbstractMCMC.Chains.iter_indices(c::MockChain) = c.iter_indices
AbstractMCMC.Chains.chain_indices(c::MockChain) = c.chain_indices
Base.size(c::MockChain) = (AbstractMCMC.Chains.niters(c), AbstractMCMC.Chains.nchains(c))
Base.keys(c::MockChain) = keys(c.data)
AbstractMCMC.Chains.get_data(c::MockChain, k::Symbol) = c.data[k]

@testset "AbstractChains interface" begin
    AbstractMCMC.Chains.test_interface(MOCK)
end

end
