using AbstractMCMC: AbstractMCMC
using AbstractPPL: AbstractPPL
using MCMCChains: Chains
using Random

"""
    Gibbs(sampler_map::NamedTuple)

A Gibbs sampler that allows for block sampling using different inference algorithms for each parameter.
"""
struct Gibbs{T<:NamedTuple} <: AbstractMCMC.AbstractSampler
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
    flatten(trace::NamedTuple)

Flatten all the values in the trace into a single vector. Variable names information is discarded.

# Examples

```jldoctest; setup = :(using AbstractMCMC: flatten)
julia> flatten((a=ones(2), b=ones(2, 2)))
6-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0

```
"""
function flatten(trace::NamedTuple)
    return reduce(vcat, vec.(values(trace)))
end

"""
    unflatten(vec::AbstractVector, variable_names::Vector{Symbol}, variable_sizes::Vector{Tuple})

Reverse operation of flatten. Reshape the vector into the original arrays using size information.

# Examples

```jldoctest; setup = :(using AbstractMCMC: unflatten)
julia> unflatten([1,2,3,4,5], (a=(2,), b=(3,)))
(a = [1, 2], b = [3, 4, 5])

julia> unflatten([1.0,2.0,3.0,4.0,5.0,6.0], (x=(2,2), y=(2,)))
(x = [1.0 3.0; 2.0 4.0], y = [5.0, 6.0])
```
"""
function unflatten(
    vec::AbstractVector, variable_names_and_sizes::NamedTuple{variable_names}
) where {variable_names}
    result = Dict{Symbol,Array}()
    start_idx = 1
    for name in variable_names
        size = variable_names_and_sizes[name]
        end_idx = start_idx + prod(size) - 1
        result[name] = reshape(vec[start_idx:end_idx], size...)
        start_idx = end_idx + 1
    end

    return NamedTuple{variable_names}(Tuple([result[name] for name in variable_names]))
end

"""
    update_trace(trace::NamedTuple, gibbs_state::GibbsState)

Update the trace with the values from the MCMC states of the sub-problems.
"""
function update_trace(trace::NamedTuple, gibbs_state::GibbsState)
    for parameter_variable in keys(gibbs_state.mcmc_states)
        sub_state = gibbs_state.mcmc_states[parameter_variable]
        sub_state_params = Base.vec(sub_state)
        unflattened_sub_state_params = unflatten(
            sub_state_params,
            NamedTuple{(parameter_variable,)}((
                gibbs_state.variable_sizes[parameter_variable],
            )),
        )
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
    sampler::Gibbs{Tsamplingmap},
    args...;
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
        sub_problem_parameters_values = NamedTuple{(parameter_variable,)}((
            initial_params[parameter_variable],
        ))

        # LogDensityProblems' `logdensity` function expects a single vector of real numbers
        # `Gibbs` stores the parameters as a named tuple, thus we need to flatten the sub_problem_parameters_values
        # and unflatten after the sampling step
        flattened_sub_problem_parameters_values = flatten(sub_problem_parameters_values)

        sub_state = last(
            AbstractMCMC.step(
                rng,
                AbstractMCMC.LogDensityModel(
                    AbstractPPL.condition(
                        logdensity_model.logdensity, conditioning_variables_values
                    ),
                ),
                sub_sampler,
                args...;
                initial_params=flattened_sub_problem_parameters_values,
                kwargs...,
            ),
        )
        (sub_state, Tuple(size(initial_params[parameter_variable])))
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
    gibbs_state::GibbsState,
    args...;
    kwargs...,
) where {Tsamplingmap}
    (; trace, mcmc_states, variable_sizes) = gibbs_state

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

        logp = LogDensityProblems.logdensity(cond_logdensity_model, sub_state)
        sub_state = (sub_state)(logp)
        sub_state = last(
            AbstractMCMC.step(
                rng, cond_logdensity_model, sub_sampler, sub_state, args...; kwargs...
            ),
        )
        trace = update_trace(trace, gibbs_state)
        sub_state
    end
    mcmc_states = NamedTuple{Tuple(model_parameter_names)}(mcmc_states)

    return GibbsTransition(trace), GibbsState(trace, mcmc_states, variable_sizes)
end
