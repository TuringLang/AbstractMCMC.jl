export Gibbs, GibbsState

"""
    AbstractMCMC.condition(model, target_varnames, global_values)

Return a new model conditioned on all variables *not* in `target_varnames`,
using the values in `global_values`.

`target_varnames` is a vector of variable identifiers. `global_values` is an
associative collection mapping identifiers to their current raw values.

Downstream packages implement this per model type. There is no default.
"""
function condition end

"""
    Gibbs(varnames => sampler, ...)

A composable Gibbs sampler built on the AbstractMCMC interface.

Each pair maps a group of variable names to the component sampler responsible
for updating those variables. All model variables must be covered by exactly
one group.

Component samplers must implement:
- `AbstractMCMC.step(rng, conditioned_model, sampler[, state])`
- `AbstractMCMC.getparams(conditioned_model, state) → Vector{<:Real}`
- `AbstractMCMC.setparams!!(conditioned_model, state, params) → state`

# Example

```julia
sampler = Gibbs(@varname(μ) => NUTS(), @varname(σ) => MH())
chain = sample(model, sampler, 1000)
```
"""
struct Gibbs{N,S} <: AbstractSampler
    varnames::N
    samplers::S

    function Gibbs(varnames::N, samplers::S) where {N<:Tuple,S<:Tuple}
        if length(varnames) != length(samplers)
            throw(ArgumentError("Number of varname groups and samplers must match."))
        end
        return new{N,S}(varnames, samplers)
    end
end

function Gibbs(pairs::Pair...)
    varnames = map(p -> _normalise_varnames(first(p)), pairs)
    samplers = map(last, pairs)
    return Gibbs(varnames, samplers)
end

_normalise_varnames(x::AbstractVector) = x
_normalise_varnames(x) = [x]

"""
    GibbsState(global_values, sub_states)

State for the `Gibbs` sampler.

- `global_values`: current raw values of all model variables (type is
  model-specific: `VarNamedTuple` for Turing, `NamedTuple` for JuliaBUGS).
- `sub_states`: tuple of component-sampler states, one per `Gibbs.samplers` entry.
"""
struct GibbsState{G,S<:Tuple}
    global_values::G
    sub_states::S
end

function step(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::Gibbs;
    initial_params=nothing,
    kwargs...,
)
    global_values = initial_params
    sub_states = _gibbs_initial_steps(
        rng, model, sampler.varnames, sampler.samplers, global_values; kwargs...
    )
    global_values = _collect_global_values(model, sampler.varnames, sampler.samplers, sub_states)
    return _build_gibbs_transition(global_values), GibbsState(global_values, sub_states)
end

function step(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::Gibbs,
    state::GibbsState;
    kwargs...,
)
    global_values, sub_states = _gibbs_sweep(
        rng, model, sampler.varnames, sampler.samplers, state.sub_states, state.global_values;
        kwargs...,
    )
    return _build_gibbs_transition(global_values), GibbsState(global_values, sub_states)
end

function _gibbs_sweep(rng, model, varname_groups, samplers, sub_states, global_values; kwargs...)
    new_sub_states = ()
    for i in eachindex(varname_groups)
        target_vars = varname_groups[i]
        spl = samplers[i]
        sub_state = sub_states[i]

        cond_model = condition(model, target_vars, global_values)
        synced_state = setparams!!(cond_model, sub_state, getparams(cond_model, sub_state))
        _, new_sub_state = step(rng, cond_model, spl, synced_state; kwargs...)

        new_params = getparams(cond_model, new_sub_state)
        global_values = _update_global_values(model, global_values, target_vars, cond_model, new_params)
        new_sub_states = (new_sub_states..., new_sub_state)
    end
    return global_values, new_sub_states
end

function _gibbs_initial_steps(rng, model, varname_groups, samplers, global_values; kwargs...)
    sub_states = ()
    for i in eachindex(varname_groups)
        target_vars = varname_groups[i]
        spl = samplers[i]

        cond_model = condition(model, target_vars, global_values)
        _, sub_state = step(rng, cond_model, spl; kwargs..., discard_sample=true)

        if global_values === nothing
            global_values = _init_global_values(model, target_vars, cond_model, sub_state)
        else
            new_params = getparams(cond_model, sub_state)
            global_values = _update_global_values(model, global_values, target_vars, cond_model, new_params)
        end
        sub_states = (sub_states..., sub_state)
    end
    return sub_states
end

"""
    AbstractMCMC._init_global_values(model, target_vars, cond_model, sub_state)

Initialise the global parameter store from the first component's initial state.
Called once only. Downstream packages implement this per model type.
"""
function _init_global_values end

"""
    AbstractMCMC._update_global_values(model, global_values, target_vars, cond_model, new_params)

Return an updated `global_values` with the target variables set to the values
encoded in `new_params`. Downstream packages implement this per model type.
"""
function _update_global_values end

function _collect_global_values(model, varname_groups, samplers, sub_states)
    global_values = nothing
    for i in eachindex(varname_groups)
        target_vars = varname_groups[i]
        sub_state = sub_states[i]
        cond_model = condition(model, target_vars, global_values)
        if global_values === nothing
            global_values = _init_global_values(model, target_vars, cond_model, sub_state)
        else
            global_values = _update_global_values(
                model, global_values, target_vars, cond_model, getparams(cond_model, sub_state)
            )
        end
    end
    return global_values
end

"""
    AbstractMCMC._build_gibbs_transition(global_values)

Build the transition object for each `step` call. Defaults to `global_values`
itself. Override to return a richer transition type.
"""
_build_gibbs_transition(global_values) = global_values
