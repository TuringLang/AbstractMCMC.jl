using LogDensityProblems, Distributions, LinearAlgebra, Random
using OrderedCollections

struct Gibbs <: AbstractMCMC.AbstractSampler
    sampler_map::OrderedDict
end

struct GibbsState
    values::NamedTuple
    states::OrderedDict
end

struct GibbsTransition
    values::NamedTuple
end

function AbstractMCMC.step(
    rng::AbstractRNG, model, sampler::Gibbs, args...; initial_params::NamedTuple, kwargs...
)
    states = OrderedDict()
    for group in keys(sampler.sampler_map)
        sampler = sampler.sampler_map[group]
        cond_val = NamedTuple{group}([initial_params[g] for g in group]...)
        trans, state = AbstractMCMC.step(
            rng, condition(model, cond_val), sampler, args...; kwargs...
        )
        states[group] = state
    end
    return GibbsTransition(initial_params), GibbsState(initial_params, states)
end

function AbstractMCMC.step(
    rng::AbstractRNG, model, sampler::Gibbs, state::GibbsState, args...; kwargs...
)
    for group in collect(keys(sampler.sampler_map))
        sampler = sampler.sampler_map[group]
        state = state.states[group]
        trans, state = AbstractMCMC.step(
            rng, condition(model, state.values[group]), sampler, state, args...; kwargs...
        )
        # TODO: what values to condition on here? stored where?
        state.states[group] = state
    end
    return nothing
end
