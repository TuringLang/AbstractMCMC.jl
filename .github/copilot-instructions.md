# AbstractMCMC.jl - GitHub Copilot Instructions

AbstractMCMC.jl provides abstract types and interfaces for Markov chain Monte Carlo methods in Julia. It is a foundational package in the TuringLang ecosystem that defines the common interface used by MCMC samplers.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup
- Ensure Julia is available: `julia --version` (requires Julia 1.6 or later)
- Navigate to project root: `cd /path/to/AbstractMCMC.jl`
- Install package dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"` 
  - Takes 30-60 seconds on first run. NEVER CANCEL. Set timeout to 120+ seconds.
  - Downloads and precompiles ~50+ Julia packages including BangBang, LogDensityProblems, Transducers

### Run Tests
- `julia --project=. -e "using Pkg; Pkg.test()"`
- Takes 2 minutes 50 seconds. NEVER CANCEL. Set timeout to 5+ minutes.
- Runs 234 tests covering sampling, steppers, transducers, and LogDensityProblems integration
- All tests must pass - any failures indicate broken functionality

### Code Formatting
- Check/fix code formatting: `julia --project=. -e "using Pkg; Pkg.add(\"JuliaFormatter\"); using JuliaFormatter; format(\".\")"`
- Takes less than 1 second after JuliaFormatter is installed  
- Uses Blue style formatting (.JuliaFormatter.toml configures this)
- ALWAYS run before committing - CI will fail without proper formatting

### Build Documentation  
- Navigate to docs: `cd docs`
- Install doc dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=joinpath(\"..\")))"` 
  - Takes 45-60 seconds. Set timeout to 120+ seconds.
- Build docs: `julia --project=. make.jl`
  - Takes 10-15 seconds. Set timeout to 30+ seconds.
- Documentation builds to docs/build/ folder

## Validation

### Basic Functionality Test
After making changes, ALWAYS validate core functionality works by running this test:
```bash
cd /path/to/AbstractMCMC.jl
julia --project=. -e "
using AbstractMCMC
using LogDensityProblems
using Random

include(\"test/utils.jl\")
Random.seed!(1234)

# Test basic MCMC sampling
model = MyModel()
sampler = MySampler()
chain = sample(model, sampler, 10; progress=false)
println(\"✓ Sample successful! Chain length: \", length(chain))
println(\"✓ First sample: a=\", chain[1].a, \", b=\", chain[1].b)
println(\"✓ Last sample: a=\", chain[end].a, \", b=\", chain[end].b)
println(\"✓ Validation test completed successfully\")
"
```
Expected output should show successful sampling with numerical values.

### CI Validation Requirements
ALWAYS run these before submitting changes:
1. `julia --project=. -e "using Pkg; Pkg.test()"` - All tests must pass
2. `julia --project=. -e "using JuliaFormatter; format(\".\")"` - Code must be formatted
3. Build documentation successfully in docs/ directory

## Repository Structure

### Source Code (`src/`)
- `AbstractMCMC.jl` - Main module file defining abstract types and exports
- `interface.jl` - Core interface definitions for step, sample functions  
- `sample.jl` - Main sampling implementation with serial/parallel support
- `stepper.jl` - Iterator interface for MCMC chains
- `transducer.jl` - Transducers.jl integration for MCMC sampling
- `logging.jl` - Progress logging and output formatting
- `logdensityproblems.jl` - LogDensityProblems.jl integration
- `samplingstats.jl` - Statistics collection during sampling

### Tests (`test/`)
- `runtests.jl` - Main test runner that includes all test files
- `utils.jl` - Test utilities with MyModel, MySampler example implementations  
- `sample.jl` - Tests for sampling functions (serial, parallel, progress logging)
- `stepper.jl` - Tests for iterator interface
- `transducer.jl` - Tests for Transducers.jl integration
- `logdensityproblems.jl` - Tests for LogDensityProblems.jl integration

### Documentation (`docs/`)
- `src/index.md` - Main documentation homepage
- `src/api.md` - User-facing API documentation
- `src/design.md` - Developer guide for implementing interfaces
- `make.jl` - Documentation build script

## Key Concepts

### Abstract Types
- `AbstractModel` - Represents a probabilistic model for inference
- `AbstractSampler` - Base type for MCMC sampling algorithms  
- `AbstractChains` - Stores parameter samples from MCMC process
- `AbstractMCMCEnsemble` - Algorithms for parallel sampling (MCMCThreads, MCMCDistributed, MCMCSerial)

### Core Interface Functions  
Every MCMC sampler implementation must define:
- `AbstractMCMC.step(rng, model, sampler, state; kwargs...)` - Single sampling step
- Optionally `AbstractMCMC.step_warmup(...)` - Warm-up sampling step

### Common Parameters
- `progress` - Enable/disable progress logging (default: true)
- `chain_type` - Output type for collected samples (default: Any)
- `num_warmup` - Number of warm-up steps before regular sampling
- `discard_initial` - Number of initial samples to discard
- `thinning` - Factor for thinning samples (keep every nth sample)

## CI Workflows

The repository has several GitHub Actions workflows:
- **CI.yml** - Main testing across Julia versions and architectures (1.6+, x64/x86, Ubuntu/Windows/macOS)  
- **Format.yml** - Checks code formatting with JuliaFormatter
- **Docs.yml** - Builds and deploys documentation
- **IntegrationTest.yml** - Tests compatibility with downstream packages (AdvancedHMC.jl, Turing.jl, etc.)

## Common Development Tasks

### Adding New Functionality
1. Implement changes in appropriate `src/` files
2. Add corresponding tests in `test/` files  
3. Update documentation in `docs/src/` if needed
4. Run validation: tests, formatting, documentation build
5. Ensure integration tests pass with downstream packages

### Debugging Issues
- Check test output for specific failure details
- Use the validation test above to isolate basic functionality
- Examine `test/utils.jl` for example implementations of interfaces
- Review existing implementations in `src/` files for patterns

### Performance Considerations
- This is an interface package - performance is mainly in downstream implementations
- Focus on API design and interface completeness rather than algorithmic performance
- Ensure abstractions don't introduce unnecessary overhead

## Package Dependencies

Key runtime dependencies:
- **BangBang.jl** - Mutating/non-mutating function interface
- **LogDensityProblems.jl** - Interface for log density functions
- **Transducers.jl** - Composable algorithms for processing data  
- **ProgressLogging.jl/TerminalLoggers.jl** - Progress reporting
- **StatsBase.jl** - Statistical computing foundation

## Troubleshooting

### Common Issues
- **Package instantiation fails**: Check internet connection for package downloads
- **Tests fail**: Ensure all dependencies installed correctly with `Pkg.instantiate()`
- **Formatting errors**: Run JuliaFormatter before committing
- **Documentation build fails**: Ensure package is added in development mode to docs environment

### Network Issues
If package downloads fail due to network restrictions:
- The package will attempt to download from GitHub and JuliaRegistries
- Allow access to github.com and pkg.julialang.org domains
- Alternatively, use offline Julia registries if available

### Integration Test Failures
If downstream packages fail in IntegrationTest.yml:
- Check if changes break SemVer compatibility
- Review breaking changes impact on AdvancedHMC.jl, AdvancedMH.jl, MCMCChains.jl, etc.
- Consider if changes require coordinated updates across TuringLang ecosystem