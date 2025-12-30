# Callbacks

AbstractMCMC provides a comprehensive callback system that allows users to execute custom code after each sampling step. This is useful for monitoring, logging, visualization, and early stopping.

## Using Callbacks

Callbacks are passed to `sample` via the `callback` keyword argument:

```julia
chain = sample(model, sampler, 1000; callback=my_callback)
```

A callback is any callable that accepts the following signature:

```julia
callback(rng, model, sampler, transition, state, iteration; kwargs...)
```

Where:
- `rng`: The random number generator used for sampling
- `model`: The model being sampled
- `sampler`: The sampler being used
- `transition`: The current sample/transition
- `state`: The current sampler state
- `iteration`: The current iteration number (1-based)
- `kwargs...`: Additional keyword arguments passed from `sample`

## Built-in Callbacks

### MultiCallback

`MultiCallback` allows combining multiple callbacks into one:

```julia
cb1 = (args...; kwargs...) -> println("Callback 1 at iteration $(args[6])")
cb2 = (args...; kwargs...) -> println("Callback 2 at iteration $(args[6])")

multi = MultiCallback(cb1, cb2)
chain = sample(model, sampler, 100; callback=multi)
```

You can also add callbacks dynamically using `push!!` from [BangBang.jl](https://github.com/JuliaFolds/BangBang.jl):

```julia
using BangBang
multi = MultiCallback()
multi = push!!(multi, cb1)
multi = push!!(multi, cb2)
```

### NameFilter

`NameFilter` is a utility for filtering parameter names by inclusion/exclusion lists:

```julia
# Only include parameters "a" and "b"
filter = NameFilter(include=["a", "b"])
filter("a")  # true
filter("c")  # false

# Exclude parameters "x" and "y"
filter = NameFilter(exclude=["x", "y"])
filter("a")  # true
filter("x")  # false

# Combined: include takes precedence, then exclude is applied
filter = NameFilter(include=["a", "b", "c"], exclude=["c"])
filter("a")  # true
filter("c")  # false

# Default: accept all
filter = NameFilter()
filter("anything")  # true
```

## TensorBoard Integration

AbstractMCMC provides built-in TensorBoard logging through a package extension. To use it, load `TensorBoardLogger` and `OnlineStats`:

```julia
using AbstractMCMC
using TensorBoardLogger, OnlineStats

# Create the callback
cb = TensorBoardCallback("tensorboard_logs/run")

# Sample with logging
chain = sample(model, sampler, 1000; callback=cb)
```

### Setting up TensorBoard

First, install TensorBoard (Python):

```sh
pip install tensorboard
```

Then start TensorBoard pointing to your log directory:

```sh
tensorboard --logdir tensorboard_logs/run
```

Navigate to `localhost:6006` in your browser to see the dashboard.

### TensorBoardCallback Options

```julia
TensorBoardCallback(
    directory::String;
    # Statistics estimator options
    stats = nothing,           # OnlineStat or Dict for custom statistics
    num_bins::Int = 100,       # Number of bins for histograms
    
    # Parameter filtering
    filter = nothing,          # Custom filter function (varname, value) -> Bool
    include = nothing,         # Only log these parameter names
    exclude = nothing,         # Don't log these parameter names
    
    # Extra statistics (log density, acceptance rate, etc.)
    include_extras::Bool = true,
    extras_filter = nothing,
    extras_include = nothing,
    extras_exclude = nothing,
    
    # Hyperparameters (logged once at start)
    include_hyperparams::Bool = false,
    hyperparams_filter = nothing,
    hyperparams_include = nothing,
    hyperparams_exclude = nothing,
    
    # Prefixes for logged values
    param_prefix::String = "",
    extras_prefix::String = "extras/",
)
```

### Custom Statistics

By default, `TensorBoardCallback` computes `Mean`, `Variance`, and `KHist` (histogram) for each parameter. You can customize this using `OnlineStats`:

```julia
using OnlineStats

# Skip first 100 warmup samples, then compute various statistics
stats = Skip(
    100,
    Series(
        Mean(), Variance(), AutoCov(10), KHist(100)
    )
)

cb = TensorBoardCallback("logs", stats)
```

### Online Statistics Wrappers

The extension provides three useful wrappers for `OnlineStat`:

#### Skip

Skip the first `b` observations before computing statistics:

```julia
# Skip first 100 warmup iterations
stat = Skip(100, Mean())
```

#### Thin

Only use every `b`-th observation:

```julia
# Use every 10th sample
stat = Thin(10, Variance())
```

#### WindowStat

Compute statistics only on the last `b` observations (sliding window):

```julia
# Stats over last 1000 samples only
stat = WindowStat(1000, Mean())
```

### Advanced Statistics Example

```julia
using OnlineStats

# Statistics combining Skip, Thin, and WindowStat
num_adapts = 100

stats = Skip(
    num_adapts,
    Series(
        # Full chain statistics
        Series(Mean(), Variance(), AutoCov(10), KHist(100)),
        # Thinned statistics (every 10th sample)
        Thin(10, Series(Mean(), Variance())),
        # Windowed statistics (last 1000 samples)
        WindowStat(1000, Series(Mean(), Variance()))
    )
)

cb = TensorBoardCallback("logs", stats)
chain = sample(model, sampler, 10000; callback=cb)
```

### Filtering Parameters

You can control which parameters are logged:

```julia
# Only log specific parameters
cb = TensorBoardCallback("logs"; include=["mu", "sigma"])

# Exclude certain parameters
cb = TensorBoardCallback("logs"; exclude=["latent_state"])

# Exclude extra statistics like log_density
cb = TensorBoardCallback("logs"; include_extras=false)

# Custom filter function
my_filter(name, value) = !startswith(name, "internal_")
cb = TensorBoardCallback("logs"; filter=my_filter)
```

## Implementing Custom Callbacks

### Simple Counter

```julia
mutable struct CounterCallback
    count::Int
end
CounterCallback() = CounterCallback(0)

function (cb::CounterCallback)(rng, model, sampler, transition, state, iteration; kwargs...)
    cb.count += 1
end

counter = CounterCallback()
chain = sample(model, sampler, 100; callback=counter)
println("Total iterations: $(counter.count)")  # 100
```

### Progress Logging

```julia
function progress_callback(rng, model, sampler, transition, state, iteration; kwargs...)
    if iteration % 100 == 0
        println("Completed iteration $iteration")
    end
end

chain = sample(model, sampler, 1000; callback=progress_callback)
```

### Combining Callbacks

```julia
counter = CounterCallback()
logger = (args...; kwargs...) -> println("Step $(args[6])")

combined = MultiCallback(counter, logger)
chain = sample(model, sampler, 100; callback=combined)
```

## Supporting TensorBoardCallback for Custom Samplers

To make your sampler compatible with `TensorBoardCallback`, implement these methods:

### params_and_values

Returns parameter names and values from a transition/state:

```julia
function AbstractMCMC.params_and_values(model::MyModel, transition::MyTransition, state; kwargs...)
    return [("param1", transition.value1), ("param2", transition.value2)]
end
```

For simple cases with unnamed parameters:

```julia
function AbstractMCMC.params_and_values(model::MyModel, state::Vector{Float64}; kwargs...)
    names = AbstractMCMC.default_param_names_for_values(state)  # ["θ[1]", "θ[2]", ...]
    return zip(names, state)
end
```

### extras

Returns additional statistics (log density, acceptance rate, etc.):

```julia
function AbstractMCMC.extras(model::MyModel, transition, state; kwargs...)
    return [
        ("log_density", transition.log_density),
        ("acceptance_rate", state.acceptance_rate),
        ("step_size", state.step_size)
    ]
end
```

### hyperparams (optional)

Returns sampler hyperparameters (logged once at the start):

```julia
function AbstractMCMC.hyperparams(model::MyModel, sampler::MySampler; kwargs...)
    return [
        "n_leapfrog" => sampler.n_leapfrog,
        "step_size" => sampler.step_size,
        "target_accept" => sampler.target_accept
    ]
end
```

### hyperparam_metrics (optional)

Returns names of metrics to associate with hyperparameters in TensorBoard:

```julia
function AbstractMCMC.hyperparam_metrics(model::MyModel, sampler::MySampler; kwargs...)
    return ["extras/log_density", "extras/acceptance_rate"]
end
```

## API Reference

### Core Callback Types

```@docs
AbstractMCMC.MultiCallback
AbstractMCMC.NameFilter
```

### Parameter Extraction Functions

```@docs
AbstractMCMC.params_and_values
AbstractMCMC.default_param_names_for_values
AbstractMCMC.extras
AbstractMCMC.hyperparams
AbstractMCMC.hyperparam_metrics
```
