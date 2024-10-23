# Proposal for a New LogDensity Function Interface

<https://github.com/TuringLang/DynamicPPL.jl/issues/691>

The goal is to design a flexible, user-friendly interface for log density functions that can handle various model operations, especially in higher-order contexts like Gibbs sampling and Bayesian workflows.

## Evaluation functions:

1. `evaluate`

## Query functions:

1. `is_parametric(model)`
2. `dimension(model)` (only defined when `is_parametric(model) == true`)
3. `is_conditioned(model)`
4. `is_fixed(model)`
5. `logjoint(model, params)`
6. `loglikelihood(model, params)`
7. `logprior(model, params)`

where `params` can be `Vector`, `NamedTuple`, `Dict`, etc.

## Transformation functions:

1. `condition(model, conditioned_vars)`
2. `fix(model, fixed_vars)`
3. `factor(model, variables_in_the_factor)`

`condition` and `factor` are similar, but `factor` effectively generates a sub-model.

## Higher-order functions:

1. `generated_quantities(model, sample, [, expr])` or `generated_quantities(model, sample, f, args...)`
   1. `generated_quantities` computes things from the sampling result.
   2. In `DynamicPPL`, this is the model's return value. For more flexibility, we should allow passing an expression or function. (Currently, users can rewrite the model definition to achieve this in `DynamicPPL`, but with limitations. We want to make this more generic.)
   3. `rand` is a special case of `generated_quantities` (when no sample is passed).
2. `predict(model, sample)`

`generated_quantities` can be implemented by `fix`ing the model on `sample` and calling `evaluate`.
`predict` can be implemented by `uncondition`ing the model on `data`, fixing it on `sample`, and calling `evaluate`.
