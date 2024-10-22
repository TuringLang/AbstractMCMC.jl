# Proposal for a New LogDensity Function Interface

## Introduction

The goal is to design a flexible and user-friendly interface for log density functions that can handle various model operations, especially in higher-order contexts such as Gibbs sampling. This interface should facilitate:

- **Conditioning**: Incorporating observed data into the model.
- **Fixing**: Fixing certain variables to specific values. (like `do` operator)
- **Generated Quantities**: Computing additional expressions or functions based on the model parameters.
- **Prediction**: Making predictions by fixing parameters and unconditioning on data.

This proposal aims to redefine the interface from the user's perspective, focusing on ease of use and extensibility beyond the traditional probabilistic programming languages (PPLs).

## Proposed Interface

Below is a proposed interface with key functionalities and their implementations.

### Core Functions

#### Check if a Model is Parametric

```julia
# Check if a log density model is parametric
function is_parametric(model::LogDensityModel) -> Bool
    ...
end
```

- **Description**: Determines if the model has a parameter space with a defined dimension.
-

#### Get the Dimension of a Parametric Model

```julia
# Get the dimension of the parameter space (only defined when is_parametric(model) is true)
function dimension(model::LogDensityModel) -> Int
    ...
end
```

- **Description**: Returns the dimension of the parameter space for parametric models.

### Log Density Computations

#### Log-Likelihood

```julia
# Compute the log-likelihood given parameters
function loglikelihood(model::LogDensityModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end
```

- **Description**: Computes the log-likelihood of the data given the model parameters.

#### Log-Prior

```julia
# Compute the log-prior given parameters
function logprior(model::LogDensityModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end
```

- **Description**: Computes the log-prior probability of the model parameters.

#### Log-Joint

```julia
# Compute the log-joint density (log-likelihood + log-prior)
function logjoint(model::LogDensityModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    return loglikelihood(model, params) + logprior(model, params)
end
```

- **Description**: Computes the total log density by summing the log-likelihood and log-prior.

### Conditioning and Fixing Variables

#### Conditioning a Model

```julia
# Condition the model on observed data
function condition(model::LogDensityModel, data::NamedTuple) -> ConditionedModel
    ...
end
```

- **Description**: Incorporates observed data into the model, returning a `ConditionedModel`.

#### Checking if a Model is Conditioned

```julia
# Check if a model is conditioned
function is_conditioned(model::LogDensityModel) -> Bool
    ...
end
```

- **Description**: Checks whether the model has been conditioned on data.

#### Fixing Variables in a Model

```julia
# Fix certain variables in the model
function fix(model::LogDensityModel, variables::NamedTuple) -> FixedModel
    ...
end
```

- **Description**: Fixes specific variables in the model to given values, returning a `FixedModel`.

#### Checking if a Model has Fixed Variables

```julia
# Check if a model has fixed variables
function is_fixed(model::LogDensityModel) -> Bool
    ...
end
```

- **Description**: Determines if any variables in the model have been fixed.

### Specialized Models

#### Conditioned Model Methods

```julia
# Log-likelihood for a conditioned model
function loglikelihood(model::ConditionedModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end

# Log-prior for a conditioned model
function logprior(model::ConditionedModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end

# Log-joint for a conditioned model
function logjoint(model::ConditionedModel, params::Union{Vector, NamedTuple, Dict}) -> Float64
    return loglikelihood(model, params) + logprior(model, params)
end
```

- **Description**: Overrides log density computations to account for the conditioned data.

#### Fixed Model Methods

```julia
# Log-likelihood for a fixed model
function loglikelihood(model::FixedModel, data::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end

# Log-prior for a fixed model
function logprior(model::FixedModel, data::Union{Vector, NamedTuple, Dict}) -> Float64
    ...
end

# Log-joint for a fixed model
function logjoint(model::FixedModel, data::Union{Vector, NamedTuple, Dict}) -> Float64
    return loglikelihood(model, data) + logprior(model, data)
end
```

- **Description**: Adjusts log density computations based on the fixed variables.

### Additional Functionalities

#### Generated Quantities

```julia
# Compute generated quantities after fixing parameters
function generated_quantities(model::LogDensityModel, fixed_vars::NamedTuple) -> NamedTuple
    ...
end
```

- **Description**: Computes additional expressions or functions based on the fixed model parameters.

#### Prediction

```julia
# Predict data based on fixed parameters
function predict(model::LogDensityModel, params::Union{Vector, NamedTuple, Dict}) -> NamedTuple
    ...
end
```

- **Description**: Generates predictions by fixing the parameters and unconditioning the data.

## Advantages of the Proposed Interface

- **Flexibility**: Allows for advanced model operations like conditioning and fixing, essential for methods like Gibbs sampling.

- **User-Centric Design**: Focuses on usability from the model user's perspective rather than the PPL implementation side.

- **Consistency**: Maintains a uniform interface for both parametric and non-parametric models, simplifying the learning curve.

## Usage Examples

## Non-Parametric Models