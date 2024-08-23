using AbstractMCMC
using Distributions
using LogDensityProblems
using OrderedCollections
using Random
using Test

include("hier_normal.jl")
# include("gmm.jl")
include("mh.jl")

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
            vi = merge(vi, unflatten(AbstractMCMC.get_params(sub_state), group))
        end
        sub_spl = spl.sampler_map[group]
        sub_state = state.states[group]
        group_complement = setdiff(keys(vi), group)
        cond_val = NamedTuple{Tuple(group_complement)}(
            Tuple([vi[g] for g in group_complement])
        )
        cond_logdensity = condition(logdensity_model.logdensity, cond_val)
        sub_state = recompute_logprob!!(
            cond_logdensity, AbstractMCMC.get_params(sub_state), sub_state
        )
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
        vi = merge(vi, unflatten(AbstractMCMC.get_params(sub_state), group))
    end
    return GibbsTransition(vi), GibbsState(vi, state.states)
end

## tests with hierarchical normal model

# generate data
N = 1000  # Number of data points
mu_true = 0.5  # True mean
tau2_true = 2.0  # True variance

# Generate data based on true parameters
x_data = rand(Normal(mu_true, sqrt(tau2_true)), N)

# Store the generated data in the HierNormal structure
hn = HierNormal((x=x_data,))

samples = sample(
    hn,
    Gibbs(
        OrderedDict(
            (:mu,) => RWMH(1),
            (:tau2,) => PriorMH(product_distribution([InverseGamma(1, 1)])),
        ),
    ),
    200000;
    initial_params=(mu=[0.0], tau2=[1.0]),
)

mu_samples = [sample.values.mu for sample in samples][40001:end]
tau2_samples = [sample.values.tau2 for sample in samples][40001:end]

mu_mean = mean(mu_samples)[1]
tau2_mean = mean(tau2_samples)[1]

@testset "hierarchical normal with gibbs" begin
    @test mu_mean ≈ mu_true atol = 0.1
    @test tau2_mean ≈ tau2_true atol = 0.3
end

## test with gmm -- too hard, doesn't converge

# gmm = GMM((; x=x))

# samples = sample(
#     gmm,
#     Gibbs(
#         OrderedDict(
#             (:z,) => PriorMH(product_distribution([Categorical([0.3, 0.7]) for _ in 1:60])),
#             (:w,) => PriorMH(Dirichlet(2, 1.0)),
#             (:μ,) => RWMH(1),
#         ),
#     ),
#     100000;
#     initial_params=(z=rand(Categorical([0.3, 0.7]), 60), μ=[-3.5, 0.5], w=[0.3, 0.7]),
# );

# z_samples = [sample.values.z for sample in samples][20001:end]
# μ_samples = [sample.values.μ for sample in samples][20001:end]
# w_samples = [sample.values.w for sample in samples][20001:end];

# # thin these samples
# z_samples = z_samples[1:100:end]
# μ_samples = μ_samples[1:100:end]
# w_samples = w_samples[1:100:end];

# mean(μ_samples)
# mean(w_samples)
