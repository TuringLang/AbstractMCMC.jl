"""
    Gibbs(sampler_map::NamedTuple)

An interface for block sampling in Markov Chain Monte Carlo (MCMC).

Gibbs sampling is a technique for dividing complex multivariate problems into simpler subproblems.
It allows different sampling methods to be applied to different parameters.
"""
struct Gibbs <: AbstractMCMC.AbstractSampler
    sampler_map::NamedTuple
    parameter_names::Tuple{Vararg{Symbol}}

    function Gibbs(sampler_map::NamedTuple)
        parameter_names = Tuple(keys(sampler_map))
        return new(sampler_map, parameter_names)
    end
end

struct GibbsState
    """
    `trace` contains the values of the values of _all_ parameters up to the last iteration.
    """
    trace::NamedTuple

    """
    `mcmc_states` maps parameters to their sampler-specific MCMC states.
    """
    mcmc_states::NamedTuple

    """
    `variable_sizes` maps parameters to their sizes.
    """
    variable_sizes::NamedTuple
end

struct GibbsTransition
    """
    Realizations of the parameters, this is considered a "sample" in the MCMC chain.
    """
    values::NamedTuple
end

"""
    flatten(trace::Union{NamedTuple,OrderedCollections.OrderedDict})

Flatten all the values in the trace into a single vector.

# Examples

```jldoctest; setup = :(using AbstractMCMC: flatten)
julia> flatten((a=[1,2], b=[3,4,5]))
[1, 2, 3, 4, 5]

julia> flatten(OrderedCollections.OrderedDict(:x=>[1.0,2.0], :y=>[3.0,4.0,5.0]))
[1.0, 2.0, 3.0, 4.0, 5.0]
```
"""
function flatten(trace::NamedTuple)
    return reduce(vcat, vec.(values(trace)))
end

"""
    unflatten(vec::AbstractVector, group_names_and_sizes::NamedTuple)

Reverse operation of flatten. Reshape the vector into the original arrays using size information.

# Examples

```jldoctest; setup = :(using AbstractMCMC: unflatten)
julia> unflatten([1,2,3,4,5], (a=(2,), b=(3,)))
(a=[1,2], b=[3,4,5])

julia> unflatten([1.0,2.0,3.0,4.0,5.0,6.0], (x=(2,2), y=(2,)))
(x=[1.0 3.0; 2.0 4.0], y=[5.0,6.0])
```
"""
function unflatten(vec::AbstractVector, variable_sizes::NamedTuple)
    result = Dict{Symbol,Array}()
    start_idx = 1
    for name in keys(variable_sizes)
        size = variable_sizes[name]
        end_idx = start_idx + prod(size) - 1
        result[name] = reshape(vec[start_idx:end_idx], size...)
        start_idx = end_idx + 1
    end

    # ensure the order of the keys is the same as the one in variable_sizes
    return NamedTuple{Tuple(keys(variable_sizes))}([
        result[name] for name in keys(variable_sizes)
    ])
end

"""
    update_trace(trace::NamedTuple, gibbs_state::GibbsState)

Update the trace with the values from the MCMC states of the sub-problems.
"""
function update_trace(trace::NamedTuple, gibbs_state::GibbsState)
    for parameter_variable in keys(gibbs_state.mcmc_states)
        sub_state = gibbs_state.mcmc_states[parameter_variable]
        trace = merge(
            trace,
            unflatten(
                vec(sub_state),
                NamedTuple{(parameter_variable,)}((
                    gibbs_state.variable_sizes[parameter_variable],
                )),
            ),
        )
    end
    return trace
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs,
    args...;
    initial_params::NamedTuple,
    kwargs...,
)
    if Set(keys(initial_params)) != Set(sampler.parameter_names)
        throw(
            ArgumentError(
                "initial_params must contain all parameters in the model, expected $(sampler.parameter_names), got $(keys(initial_params))",
            ),
        )
    end

    mcmc_states = Dict{Symbol,Any}()
    variable_sizes = Dict{Symbol,Tuple}()
    for parameter_variable in sampler.parameter_names
        sub_sampler = sampler.sampler_map[parameter_variable]

        variables_to_be_conditioned_on = setdiff(
            sampler.parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([initial_params[g] for g in variables_to_be_conditioned_on])
        )
        sub_problem_parameters_values = NamedTuple{(parameter_variable,)}((
            initial_params[parameter_variable],
        ))

        # LogDensityProblems' `logdensity` function expects a single vector of real numbers
        # `Gibbs` stores the parameters as a named tuple, thus we need to flatten the sub_problem_parameters_values
        # and unflatten after the sampling step
        variable_sizes[parameter_variable] = Tuple(size(initial_params[parameter_variable]))
        flattened_sub_problem_parameters_values = flatten(sub_problem_parameters_values)

        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(
                    AbstractMCMC.condition(
                        logdensity_model.logdensity, conditioning_variables_values
                    ),
                ),
                sub_sampler,
                args...;
                initial_params=flattened_sub_problem_parameters_values,
                kwargs...,
            ),
        )
        mcmc_states[parameter_variable] = sub_state
    end

    gibbs_state = GibbsState(
        initial_params, NamedTuple(mcmc_states), NamedTuple(variable_sizes)
    )
    trace = update_trace(NamedTuple(), gibbs_state)
    return GibbsTransition(trace), gibbs_state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs,
    gibbs_state::GibbsState,
    args...;
    kwargs...,
)
    trace = gibbs_state.trace
    mcmc_states = gibbs_state.mcmc_states
    variable_sizes = gibbs_state.variable_sizes
    
    mcmc_states_dict = Dict(
        keys(mcmc_states) .=> [mcmc_states[k] for k in keys(mcmc_states)]
    )
    for parameter_variable in sampler.parameter_names
        sub_sampler = sampler.sampler_map[parameter_variable]
        sub_state = mcmc_states[parameter_variable]
        variables_to_be_conditioned_on = setdiff(
            sampler.parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([trace[g] for g in variables_to_be_conditioned_on])
        )
        cond_logdensity = AbstractMCMC.condition(
            logdensity_model.logdensity, conditioning_variables_values
        )

        # recompute the logdensity stored in the mcmc state, because the values might have been updated in other sub-problems
        updated_log_prob = LogDensityProblems.logdensity(cond_logdensity, sub_state)

        if !hasproperty(sub_state, :logp)
            error(
                "$(typeof(sub_state)) does not have a `:logp` field, which is required by Gibbs sampling",
            )
        end
        sub_state = BangBang.setproperty!!(sub_state, :logp, updated_log_prob)

        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(cond_logdensity),
                sub_sampler,
                sub_state,
                args...;
                kwargs...,
            ),
        )
        mcmc_states_dict[parameter_variable] = sub_state
        trace = update_trace(trace, gibbs_state)
    end

    mcmc_states = NamedTuple{Tuple(keys(mcmc_states_dict))}(
        Tuple([mcmc_states_dict[k] for k in keys(mcmc_states_dict)])
    )
    return GibbsTransition(trace), GibbsState(trace, mcmc_states, variable_sizes)
end
