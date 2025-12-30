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

AbstractMCMC provides built-in TensorBoard logging through a package extension. `TensorBoardCallback` is a wrapper around `Base.CoreLogging.AbstractLogger` which can be used to create a callback compatible with `AbstractMCMC.sample`. This allows you to visualize your MCMC sampling progress in real-time using TensorBoard's web interface.

### Basic Usage

```julia
using Turing, AbstractMCMC
using TensorBoardLogger, OnlineStats

@model function demo(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, √s)
    for i in eachindex(x)
        x[i] ~ Normal(m, √s)
    end
end

xs = randn(100) .+ 1;
model = demo(xs);

# Number of MCMC samples/steps
num_samples = 10_000
num_adapts = 100

# Sampling algorithm to use
alg = NUTS(num_adapts, 0.65)

# Create the callback
cb = TensorBoardCallback("tensorboard_logs/run")

# Sample with logging
chain = sample(model, alg, num_samples; callback = cb)
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

Navigate to `localhost:6006` in your browser to see the dashboard. You'll see real-time plots of your parameter distributions, histograms, and other statistics as sampling progresses. In particular, note the "Distributions" and "Histograms" tabs which show the marginal distributions of your parameters.

![TensorBoard Time Series Tab](assets/tensorboard_demo_time-series_screen.png)

*The Time Series tab provides detailed traces of parameter values throughout the sampling process.*

![TensorBoard Scalars Tab](assets/tensorboard_demo_scalars_screen.png)

*The Scalars tab shows time series of parameter values and statistics over the sampling iterations.*

![TensorBoard Distributions Tab](assets/tensorboard_demo_distributions_screen.png)

*The Distributions tab displays the marginal distributions of each parameter.*

![TensorBoard Histograms Tab](assets/tensorboard_demo_histograms_screen.png)

*The Histograms tab shows the evolution of parameter distributions over time.*


### TensorBoardCallback Options

```@docs
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

See the [`OnlineStats.jl` documentation](https://joshday.github.io/OnlineStats.jl/latest/) for more on the different statistics, with the exception of [`Thin`](@ref), [`Skip`](@ref) and [`WindowStat`](@ref) which are implemented in this package.

!!! note
    OnlineStat objects are stateful and accumulate data over time. Be careful not to reuse the same stat object in multiple places, as this can lead to incorrect results. For example, avoid:
    
    ```julia
    s = AutoCov(5)
    stat = Series(s, s)
    # => 10 samples but `n=20` since we've called `fit!` twice for each observation
    fit!(stat, randn(10))
    ```
    
    Instead, create separate instances:
    
    ```julia
    stat = Series(AutoCov(5), AutoCov(5))
    # => 10 samples AND `n=10`; great!
    fit!(stat, randn(10))
    ```

For custom statistics, implement the OnlineStats interface by defining `OnlineStats.fit!` and `OnlineStats.value` methods. By default, an `OnlineStat` is passed to TensorBoard by simply calling `OnlineStats.value(stat)`. If you want to customize how a stat is passed to TensorBoard, you need to overload `TensorBoardLogger.preprocess(name, stat, data)` accordingly. See the [OnlineStats.jl documentation](https://joshday.github.io/OnlineStats.jl/latest/) for more details.

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

Tada! Now you should be seeing waaaay more interesting statistics in your TensorBoard dashboard.

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

Logged statistics follow a naming convention of `$variable_name/...` where `$variable_name` refers to the name of the variable in your model.

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

!!! note
    The `params_and_values(model, sampler, transition, state; kwargs...)` method is not usually overloaded, but it can sometimes be useful for defining more complex behaviors.

### `params_and_values`

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

Sometimes the user might pass parameter names as a keyword argument, and you might want to support that:

```julia
function AbstractMCMC.params_and_values(model::MyModel, state::Vector{Float64}; param_names=nothing, kwargs...)
    param_names = isnothing(param_names) ? AbstractMCMC.default_param_names_for_values(state) : param_names
    return zip(param_names, state)
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
