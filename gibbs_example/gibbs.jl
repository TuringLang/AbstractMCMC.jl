using AbstractMCMC
using LogDensityProblems, Distributions, LinearAlgebra, Random
using OrderedCollections

##

# TODO: introduce some kind of parameter format, for instance, a flattened vector
# then define some kind of function to transform the flattened vector into model's representation

struct Gibbs <: AbstractMCMC.AbstractSampler
    sampler_map::OrderedDict
end

struct GibbsState
    vi::NamedTuple
    states::OrderedDict
end

struct GibbsTransition
    values::NamedTuple
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    spl::Gibbs,
    args...;
    initial_params::NamedTuple,
    kwargs...,
)
    states = OrderedDict()
    for group in keys(spl.sampler_map)
        sub_spl = spl.sampler_map[group]

        vars_to_be_conditioned_on = setdiff(keys(initial_params), group)
        cond_val = NamedTuple{Tuple(vars_to_be_conditioned_on)}(
            Tuple([initial_params[g] for g in vars_to_be_conditioned_on])
        )
        params_val = NamedTuple{Tuple(group)}(Tuple([initial_params[g] for g in group]))
        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(
                    condition(logdensity_model.logdensity, cond_val)
                ),
                sub_spl,
                args...;
                initial_params=flatten(params_val),
                kwargs...,
            ),
        )
        states[group] = sub_state
    end
    return GibbsTransition(initial_params), GibbsState(initial_params, states)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    spl::Gibbs,
    state::GibbsState,
    args...;
    kwargs...,
)
    vi = state.vi
    for group in keys(spl.sampler_map)
        for (group, sub_state) in state.states
            vi = merge(vi, unflatten(getparams(sub_state), group))
        end
        sub_spl = spl.sampler_map[group]
        sub_state = state.states[group]
        group_complement = setdiff(keys(vi), group)
        cond_val = NamedTuple{Tuple(group_complement)}(
            Tuple([vi[g] for g in group_complement])
        )
        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(
                    condition(logdensity_model.logdensity, cond_val)
                ),
                sub_spl,
                sub_state,
                args...;
                kwargs...,
            ),
        )
        state.states[group] = sub_state
    end
    for sub_state in values(state.states)
        vi = merge(vi, getparams(sub_state))
    end
    return GibbsTransition(vi), GibbsState(vi, state.states)
end

## tests

gmm = GMM((; x=x))

samples = sample(
    gmm,
    Gibbs(
        OrderedDict(
            (:z,) => PriorMH(product_distribution([Categorical([0.3, 0.7]) for _ in 1:60])),
            (:w,) => PriorMH(Dirichlet(2, 1.0)),
            (:μ, :w) => RWMH(1),
        ),
    ),
    10000;
    initial_params=(z=rand(Categorical([0.3, 0.7]), 60), μ=[0.0, 1.0], w=[0.3, 0.7]),
)
