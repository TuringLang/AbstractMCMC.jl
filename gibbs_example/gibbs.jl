using AbstractMCMC
using LogDensityProblems, Distributions, LinearAlgebra, Random
using OrderedCollections

##

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
        cond_logdensity = condition(logdensity_model.logdensity, cond_val)
        sub_state = recompute_logprob!!(cond_logdensity, getparams(sub_state), sub_state)
        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(cond_logdensity),
                sub_spl,
                sub_state,
                args...;
                kwargs...,
            ),
        )
        state.states[group] = sub_state
    end
    for (group, sub_state) in state.states
        vi = merge(vi, unflatten(getparams(sub_state), group))
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
            (:μ,) => RWMH(1),
        ),
    ),
    100000;
    initial_params=(z=rand(Categorical([0.3, 0.7]), 60), μ=[0.0, 1.0], w=[0.3, 0.7]),
);

z_samples = [sample.values.z for sample in samples][20001:end]
μ_samples = [sample.values.μ for sample in samples][20001:end]
w_samples = [sample.values.w for sample in samples][20001:end]

mean(μ_samples)
mean(w_samples)
