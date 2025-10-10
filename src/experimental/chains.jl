module Chains

using AbstractMCMC: AbstractChains

"""
    AbstractMCMC.Chains.get_data(chn, key)

Obtain the data associated with `key` from the `AbstractChain` object `chn`.

This function should return an `AbstractMatrix` where the rows correspond to iterations and
columns correspond to chains.
"""
function get_data end

"""
    AbstractMCMC.Chains.iter_indices(chn)

Obtain the indices of each iteration for the `AbstractChains` object `chn`.

This function should return an `AbstractVector{<:Integer}`.
"""
function iter_indices end

"""
    AbstractMCMC.Chains.chain_indices(chn)

Obtain the indices of each chain in the `AbstractChains` object `chn`.

This function should return an `AbstractVector{<:Integer}`.
"""
function chain_indices end

"""
    AbstractMCMC.Chains.niters(chn)

Obtain the number of iterations in the `AbstractChains` object `chn`.

The default implementation calculates the length of `AbstractChains.iter_indices(chn)`. You
can define your own method if you have a more efficient way of obtaining this information.
"""
niters(c::AbstractChains) = length(iter_indices(c))

"""
    AbstractMCMC.Chains.nchains(chn)

Obtain the number of chains in the `AbstractChains` object `chn`.

The default implementation calculates the length of `AbstractChains.chain_indices(chn)`. You
can define your own method if you have a more efficient way of obtaining this information.
"""
nchains(c::AbstractChains) = length(chain_indices(c))

"""
    AbstractMCMC.Chains.test_interface(chn)

Test that the `AbstractChains` object `chn` implements the required interface.
"""
function test_interface end # Extended in TestExt

# Plotting functions; to be extended by individual chain libraries
function autocorplot end
function autocorplot! end
function energyplot end
function energyplot! end
function forestplot end
function forestplot! end
function meanplot end
function meanplot! end
function mixeddensity end
function mixeddensity! end
function ppcplot end
function ppcplot! end
function ridgelineplot end
function ridgelineplot! end
function traceplot end
function traceplot! end
# Note that other functions are provided by other libraries. In particular:
# Plots.histogram
# StatsPlots.density
# StatsPlots.cornerplot

end # AbstractMCMC.Chains
