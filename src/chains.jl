"""
    SamplingOutput(samples; iterations=1:size(samples, 1), sampling_stats, sampler_states)

Store a matrix of samples and their chain-level metadata.

Rows are iterations and columns are chains. Each entry holds a draw, including any
recorded sampler statistics. The entry type is parametric: for example, a sampler can use
a `NamedTuple` or its own type for parameters and statistics. `iterations` records the
iteration indices before thinning.
`sampling_stats` contains one `SamplingStats` or `missing` per chain, and `sampler_states`
contains one saved state or `missing` per chain. Both default to `missing`; sampler states
are retained only when sampling with `save_state=true`.
Specialize `Base.show(io::IO, ::MIME"text/plain", ::SamplingOutput{<:MyDraw})` for custom display.
"""
struct SamplingOutput{T,I<:AbstractRange{<:Integer},S<:Union{SamplingStats,Missing},L} <:
       AbstractChains
    samples::Matrix{T}
    iterations::I
    sampling_stats::Vector{S}
    sampler_states::Vector{L}

    function SamplingOutput(
        samples::Matrix{T};
        iterations::I=1:size(samples, 1),
        sampling_stats::Vector{S}=fill(missing, size(samples, 2)),
        sampler_states::Vector{L}=fill(missing, size(samples, 2)),
    ) where {T,I<:AbstractRange{<:Integer},S<:Union{SamplingStats,Missing},L}
        length(iterations) == size(samples, 1) ||
            throw(DimensionMismatch("one iteration index is required per sample row"))
        Base.step(iterations) > 0 || throw(ArgumentError("iteration indices must increase"))
        length(sampling_stats) == length(sampler_states) == size(samples, 2) || throw(
            DimensionMismatch(
                "one sampling-statistics and sampler-state entry is required per chain"
            ),
        )
        return new{T,I,S,L}(samples, iterations, sampling_stats, sampler_states)
    end
end

Base.size(chain::SamplingOutput, args...) = size(chain.samples, args...)
Base.getindex(chain::SamplingOutput, args...) = getindex(chain.samples, args...)
Base.firstindex(chain::SamplingOutput, args...) = firstindex(chain.samples, args...)
Base.lastindex(chain::SamplingOutput, args...) = lastindex(chain.samples, args...)

function Base.show(io::IO, ::MIME"text/plain", output::SamplingOutput)
    n, m = size(output)
    times = [round(s.duration; sigdigits=3) for s in skipmissing(output.sampling_stats)]
    timing = isempty(times) ? missing : (min=minimum(times), max=maximum(times))
    chains = m == 1 ? "chain" : "chains"
    println(io, "SamplingOutput: $m $chains, $n draws each")
    discard_initial = first(output.iterations) - 1
    thinning = Base.step(output.iterations)
    println(io, "  Total draws:    $(n * m) ", (; discard_initial, thinning))
    println(io, "  Time/chain (s): ", timing)
    println(io)
    return printstyled(
        io,
        "For MCMC diagnostics and plotting, use FlexiChains, MCMCChains, or ArviZ.";
        color=:light_black,
    )
end

function _bundle_samples(
    samples::AbstractVecOrMat,
    ::AbstractModel,
    ::AbstractSampler,
    state,
    ::Type{C};
    save_state=false,
    stats=missing,
    discard_initial=0,
    thinning=1,
    kwargs...,
) where {C<:SamplingOutput}
    samples = samples isa Array ? samples : collect(samples)
    samples = samples isa Vector ? reshape(samples, :, 1) : samples
    return from_samples(
        C,
        samples;
        iterations=range(discard_initial + 1; step=thinning, length=size(samples, 1)),
        sampling_stats=fill(stats, size(samples, 2)),
        sampler_states=fill(save_state ? state : missing, size(samples, 2)),
    )
end

chainsstack(chains::AbstractVector{<:SamplingOutput}) = chainscat(chains...)

function chainscat(chain::SamplingOutput, chains::SamplingOutput...)
    all(c -> c.iterations == chain.iterations, chains) ||
        throw(ArgumentError("chains must have matching iteration indices"))
    all_chains = (chain, chains...)
    return SamplingOutput(
        hcat(map(c -> c.samples, all_chains)...);
        iterations=chain.iterations,
        sampling_stats=vcat(map(c -> c.sampling_stats, all_chains)...),
        sampler_states=vcat(map(c -> c.sampler_states, all_chains)...),
    )
end

function to_samples(::Type{T}, chain::SamplingOutput) where {T}
    return convert(Matrix{T}, chain.samples)
end

from_samples(::Type{SamplingOutput}, s::Matrix; kw...) = SamplingOutput(s; kw...)
function from_samples(::Type{C}, samples::Matrix; kw...) where {T,C<:SamplingOutput{T}}
    chain = SamplingOutput(convert(Matrix{T}, samples); kw...)
    chain isa C || throw(ArgumentError("samples and metadata cannot construct $C"))
    return chain
end
