# Callbacks

AbstractMCMC provides a unified callback API for monitoring and logging MCMC sampling.

## Basic Usage

The `mcmc_callback` function is the main entry point for creating callbacks:

```julia
using AbstractMCMC

# Simple callback with a function
cb = mcmc_callback() do rng, model, sampler, transition, state, iteration
    println("Iteration: $iteration")
end

chain = sample(model, sampler, 1000; callback=cb)
```

## Combining Multiple Callbacks

Pass multiple callbacks to `mcmc_callback` to combine them:

```julia
cb1 = (args...; kwargs...) -> println("Callback 1")
cb2 = (args...; kwargs...) -> println("Callback 2")

cb = mcmc_callback(cb1, cb2)
```

## TensorBoard Logging

TensorBoard logging requires **both** `TensorBoardLogger` and `OnlineStats`:

```julia
using AbstractMCMC
using TensorBoardLogger
using OnlineStats

# Basic TensorBoard logging
cb = mcmc_callback(logger=:TBLogger, logdir="runs/experiment1")

# Or just provide logdir (defaults to TBLogger)
cb = mcmc_callback(logdir="runs/experiment1")
```

!!! note
    If you use TensorBoard logging without loading `OnlineStats`, you will see:
    `"TensorBoard logging requires both TensorBoardLogger and OnlineStats."`

### Custom Statistics

Use the `stats` argument to specify which statistics to collect. Statistics must be from `OnlineStats`:

```julia
using OnlineStats

cb = mcmc_callback(
    logdir="runs/custom_stats",
    stats=(Mean(), Variance(), KHist(50)),
)
```

### Stats Processing Options

Control how samples are processed before computing statistics with `stats_options`:

```julia
cb = mcmc_callback(
    logdir="runs/processed",
    stats_options=(
        skip=100,    # Skip first 100 samples (burn-in)
        thin=5,      # Use every 5th sample
        window=1000, # Rolling window of 1000 samples
    ),
)
```

Options merge with defaults, so you only need to specify what you want to change:

```julia
# Only change thin, skip and window use defaults (0 and typemax(Int))
cb = mcmc_callback(logdir="runs/exp", stats_options=(thin=10,))
```

### Name Filtering

Use `name_filter` to control which parameters and statistics are logged:

```julia
cb = mcmc_callback(
    logdir="runs/filtered",
    name_filter=(
        include=["mu", "sigma"],  # Only log these parameters
        exclude=["_internal"],     # Exclude matching names
        extras=true,               # Include extra stats (log density, etc.)
        hyperparams=true,          # Include hyperparameters
    ),
)
```

### Complete Example

```julia
using AbstractMCMC
using TensorBoardLogger
using OnlineStats

cb = mcmc_callback(
    logdir="runs/full_example",
    stats=(Mean(), Variance(), KHist(100)),
    stats_options=(skip=50, thin=2),
    name_filter=(
        exclude=["_internal"],
        extras=true,
        hyperparams=true,
    ),
)

chain = sample(model, sampler, 10000; callback=cb)
```

## API Reference

### Main Functions

```@docs
mcmc_callback
```

### Internal Types

These types are used internally but can be accessed via `AbstractMCMC.TypeName`:

```@docs
AbstractMCMC.Callback
AbstractMCMC.MultiCallback
AbstractMCMC.NameFilter
```

!!! note
    `Skip`, `Thin`, and `WindowStat` are defined in the OnlineStats extension and are applied
    automatically based on `stats_options`. Users don't interact with them directly.

## Default Values

### stats_options defaults

| Option   | Default        | Description                    |
|----------|----------------|--------------------------------|
| `skip`   | `0`            | Skip first n samples (burn-in) |
| `thin`   | `0`            | Use every nth sample (0=all)   |
| `window` | `typemax(Int)` | Window size for rolling stats  |

### name_filter defaults

| Option       | Default    | Description                      |
|--------------|------------|----------------------------------|
| `include`    | `String[]` | Only log these (empty=all)       |
| `exclude`    | `String[]` | Don't log these                  |
| `extras`     | `false`    | Include extra stats              |
| `hyperparams`| `false`    | Include hyperparameters          |

## Implementing Custom Callbacks

Any callable with the following signature can be used as a callback:

```julia
function my_callback(rng, model, sampler, transition, state, iteration; kwargs...)
    # Your callback logic here
end
```

### Extracting Information

AbstractMCMC provides functions to extract information from samplers:

```julia
# Get parameter names and values
for (name, value) in AbstractMCMC.params_and_values(model, sampler, transition, state)
    println("$name = $value")
end

# Get extra statistics (log density, etc.)
for (name, value) in AbstractMCMC.extras(model, sampler, transition, state)
    println("$name = $value")
end

# Get hyperparameters (for first iteration)
for (name, value) in AbstractMCMC.hyperparams(model, sampler, state)
    println("$name = $value")
end
```

Samplers can override these methods to provide custom information extraction.
