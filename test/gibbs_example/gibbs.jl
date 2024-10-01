using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL
using BangBang: constructorof
using Random

struct Gibbs{T<:NamedTuple} <: AbstractMCMC.AbstractSampler
    "Maps variables to their samplers."
    sampler_map::T
end

struct GibbsState{TraceNT<:NamedTuple,StateNT<:NamedTuple,SizeNT<:NamedTuple}
    "Contains the values of all parameters up to the last iteration."
    trace::TraceNT

    "Maps parameters to their sampler-specific MCMC states."
    mcmc_states::StateNT

    "Maps parameters to their sizes."
    variable_sizes::SizeNT
end

struct GibbsTransition{ValuesNT<:NamedTuple}
    "Realizations of the parameters, this is considered a \"sample\" in the MCMC chain."
    values::ValuesNT
end

"""
    update_trace(trace::NamedTuple, gibbs_state::GibbsState)

Update the trace with the values from the MCMC states of the sub-problems.
"""
function update_trace(
    trace::NamedTuple{trace_names}, gibbs_state::GibbsState{TraceNT,StateNT,SizeNT}
) where {trace_names,TraceNT,StateNT,SizeNT}
    for parameter_variable in fieldnames(StateNT)
        sub_state = gibbs_state.mcmc_states[parameter_variable]
        sub_state_params_values = Base.vec(sub_state)
        reshaped_sub_state_params_values = reshape(
            sub_state_params_values, gibbs_state.variable_sizes[parameter_variable]
        )
        unflattened_sub_state_params = NamedTuple{(parameter_variable,)}((
            reshaped_sub_state_params_values,
        ))
        trace = merge(trace, unflattened_sub_state_params)
    end
    return trace
end

function error_if_not_fully_initialized(
    initial_params::NamedTuple{ParamNames}, sampler::Gibbs{<:NamedTuple{SamplerNames}}
) where {ParamNames,SamplerNames}
    if Set(ParamNames) != Set(SamplerNames)
        throw(
            ArgumentError(
                "initial_params must contain all parameters in the model, expected $(SamplerNames), got $(ParamNames)",
            ),
        )
    end
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs{Tsamplingmap};
    initial_params::NamedTuple,
    kwargs...,
) where {Tsamplingmap}
    error_if_not_fully_initialized(initial_params, sampler)

    model_parameter_names = fieldnames(Tsamplingmap)
    results = map(model_parameter_names) do parameter_variable
        sub_sampler = sampler.sampler_map[parameter_variable]

        variables_to_be_conditioned_on = setdiff(
            model_parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([initial_params[g] for g in variables_to_be_conditioned_on])
        )

        # LogDensityProblems' `logdensity` function expects a single vector of real numbers
        # `Gibbs` stores the parameters as a named tuple, thus we need to flatten the sub_problem_parameters_values
        # and unflatten after the sampling step
        flattened_sub_problem_parameters_values = vec(initial_params[parameter_variable])

        sub_logdensity_model = AbstractMCMC.LogDensityModel(
            AbstractPPL.condition(
                logdensity_model.logdensity, conditioning_variables_values
            ),
        )
        sub_state = last(
            AbstractMCMC.step(
                rng,
                sub_logdensity_model,
                sub_sampler;
                initial_params=flattened_sub_problem_parameters_values,
                kwargs...,
            ),
        )
        (sub_state, size(initial_params[parameter_variable]))
    end

    mcmc_states_tuple = first.(results)
    variable_sizes_tuple = last.(results)

    gibbs_state = GibbsState(
        initial_params,
        NamedTuple{Tuple(model_parameter_names)}(mcmc_states_tuple),
        NamedTuple{Tuple(model_parameter_names)}(variable_sizes_tuple),
    )

    trace = update_trace(NamedTuple(), gibbs_state)
    return GibbsTransition(trace), gibbs_state
end

# subsequent steps
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensity_model::AbstractMCMC.LogDensityModel,
    sampler::Gibbs{Tsamplingmap},
    gibbs_state::GibbsState;
    kwargs...,
) where {Tsamplingmap}
    trace = gibbs_state.trace
    mcmc_states = gibbs_state.mcmc_states
    variable_sizes = gibbs_state.variable_sizes

    model_parameter_names = fieldnames(Tsamplingmap)
    mcmc_states = map(model_parameter_names) do parameter_variable
        sub_sampler = sampler.sampler_map[parameter_variable]
        sub_state = mcmc_states[parameter_variable]
        variables_to_be_conditioned_on = setdiff(
            model_parameter_names, (parameter_variable,)
        )
        conditioning_variables_values = NamedTuple{Tuple(variables_to_be_conditioned_on)}(
            Tuple([trace[g] for g in variables_to_be_conditioned_on])
        )
        cond_logdensity = AbstractPPL.condition(
            logdensity_model.logdensity, conditioning_variables_values
        )
        cond_logdensity_model = AbstractMCMC.LogDensityModel(cond_logdensity)

        logp = LogDensityProblems.logdensity(
            cond_logdensity_model, sub_state; recompute_logp=true
        )
        sub_state = constructorof(typeof(sub_state))(sub_state, logp)
        sub_state = last(
            AbstractMCMC.step(
                rng, cond_logdensity_model, sub_sampler, sub_state; kwargs...
            ),
        )
        trace = update_trace(trace, gibbs_state)
        sub_state
    end
    mcmc_states = NamedTuple{Tuple(model_parameter_names)}(mcmc_states)

    return GibbsTransition(trace), GibbsState(trace, mcmc_states, variable_sizes)
end
