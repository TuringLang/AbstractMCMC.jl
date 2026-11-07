"""
    SamplingOutput(samples; iterations=1:size(samples, 1), sampling_stats, sampler_states)

Store a matrix of samples and their chain-level metadata.

Rows are iterations and columns are chains. Each entry holds a draw, including any
recorded sampler statistics. The entry type is parametric: for example, a sampler can use
a `NamedTuple` or its own type for parameters and statistics. `iterations` records the
iteration indices before thinning.
Two-index selections with increasing iteration indices keep a `SamplingOutput` unless both indices are scalars.
Rows and columns may be integers, ranges, integer vectors, Boolean masks, or `:`.
Iteration indices may be an increasing vector for uneven selections.
Reversed or repeated draw selections throw an `ArgumentError`.
Selections share the sample matrix through a view.
Sampling statistics and saved states still describe the original runs.
`sampling_stats` contains one `SamplingStats` or `missing` per chain, and `sampler_states`
contains one saved state or `missing` per chain. Both default to `missing`; sampler states
are retained only when sampling with `save_state=true`.
Specialize `Base.show(io::IO, ::MIME"text/plain", ::SamplingOutput{<:MyDraw})` for custom display.
"""
struct SamplingOutput{
    T,I<:AbstractVector{<:Integer},S<:Union{SamplingStats,Missing},L,A<:AbstractMatrix{T}
} <: AbstractChains
    samples::A
    iterations::I
    sampling_stats::Vector{S}
    sampler_states::Vector{L}

    function SamplingOutput(
        samples::A;
        iterations::I=1:size(samples, 1),
        sampling_stats::Vector{S}=fill(missing, size(samples, 2)),
        sampler_states::Vector{L}=fill(missing, size(samples, 2)),
    ) where {
        T,
        I<:AbstractVector{<:Integer},
        S<:Union{SamplingStats,Missing},
        L,
        A<:AbstractMatrix{T},
    }
        Base.require_one_based_indexing(samples, iterations)
        length(iterations) == size(samples, 1) ||
            throw(DimensionMismatch("one iteration index is required per sample row"))
        _increasing_iterations(iterations) ||
            throw(ArgumentError("iteration indices must increase"))
        length(sampling_stats) == length(sampler_states) == size(samples, 2) || throw(
            DimensionMismatch(
                "one sampling-statistics and sampler-state entry is required per chain"
            ),
        )
        return new{T,I,S,L,A}(samples, iterations, sampling_stats, sampler_states)
    end
end

function _increasing_iterations(iterations::AbstractRange)
    return isempty(iterations) || Base.step(iterations) > 0
end
_increasing_iterations(iterations::AbstractVector) = issorted(iterations; lt=(<=))

Base.size(chain::SamplingOutput, args...) = size(chain.samples, args...)
Base.getindex(chain::SamplingOutput, args...) = getindex(chain.samples, args...)
function Base.getindex(output::SamplingOutput, row::Integer, column::Integer)
    return output.samples[row, column]
end
function Base.getindex(
    output::SamplingOutput,
    rows::Union{Colon,Integer,AbstractVector{<:Integer}},
    columns::Union{Colon,Integer,AbstractVector{<:Integer}},
)
    indices = Base.to_indices(output.samples, (rows, columns))
    checkbounds(output.samples, indices...)
    rows, columns = indices
    rows = rows isa Integer ? (rows:rows) : copy(rows)
    columns = columns isa Integer ? (columns:columns) : copy(columns)
    samples = view(output.samples, rows, columns)
    iterations = output.iterations[rows]
    return SamplingOutput(
        samples;
        iterations,
        sampling_stats=output.sampling_stats[columns],
        sampler_states=output.sampler_states[columns],
    )
end
Base.firstindex(chain::SamplingOutput, args...) = firstindex(chain.samples, args...)
Base.lastindex(chain::SamplingOutput, args...) = lastindex(chain.samples, args...)

function Base.show(io::IO, ::MIME"text/plain", output::SamplingOutput)
    n, m = size(output)
    times = [round(s.duration; sigdigits=3) for s in skipmissing(output.sampling_stats)]
    timing = isempty(times) ? missing : (min=minimum(times), max=maximum(times))
    chains = m == 1 ? "chain" : "chains"
    println(io, "SamplingOutput: $m $chains, $n draws each")
    discard_initial = isempty(output.iterations) ? missing : first(output.iterations) - 1
    thinning = if output.iterations isa AbstractRange
        Base.step(output.iterations)
    elseif length(output.iterations) > 1
        deltas = diff(output.iterations)
        all(==(first(deltas)), deltas) ? first(deltas) : missing
    else
        missing
    end
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

from_samples(::Type{SamplingOutput}, s::AbstractMatrix; kw...) = SamplingOutput(s; kw...)
function from_samples(
    ::Type{C}, samples::AbstractMatrix; kw...
) where {T,C<:SamplingOutput{T}}
    chain = SamplingOutput(convert(AbstractMatrix{T}, samples); kw...)
    chain isa C || throw(ArgumentError("samples and metadata cannot construct $C"))
    return chain
end
