# History

## 5.17.1

- Chains resumed from `initial_state` now take exactly `num_warmup` warm-up steps. Previously they took `num_warmup + 1`, including one when `num_warmup = 0` ([#220](https://github.com/TuringLang/AbstractMCMC.jl/issues/220)).

## 5.17.0

- Add `SamplingOutput` to retain draws, iteration indices, timing, and optional sampler states in a common container that chain packages can convert to their own formats ([#218](https://github.com/TuringLang/AbstractMCMC.jl/pull/218)).
