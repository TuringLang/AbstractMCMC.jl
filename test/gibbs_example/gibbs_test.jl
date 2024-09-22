include("gibbs.jl")
include("mh.jl")
# include("gmm.jl")
include("hier_normal.jl")

@testset "hierarchical normal with gibbs" begin
    # generate data
    N = 1000  # Number of data points
    mu_true = 5  # True mean
    tau2_true = 2.0  # True variance
    x_data = rand(Distributions.Normal(mu_true, sqrt(tau2_true)), N)

    # Store the generated data in the HierNormal structure
    hn = HierNormal((x=x_data,))

    samples = sample(
        hn,
        Gibbs((
            mu=RandomWalkMH(0.3),
            tau2=IndependentMH(product_distribution([InverseGamma(1, 1)])),
        )),
        200000;
        initial_params=(mu=[0.0], tau2=[1.0]),
    )

    warmup = 40000
    thin = 10
    thinned_samples = samples[(warmup + 1):thin:end]
    mu_samples = [sample.values.mu for sample in thinned_samples]
    tau2_samples = [sample.values.tau2 for sample in thinned_samples]

    mu_mean = only(mean(mu_samples))
    tau2_mean = only(mean(tau2_samples))

    @test mu_mean ≈ mu_true atol = 0.1
    @test tau2_mean ≈ tau2_true atol = 0.3
end

# This is too difficult to sample, disable for now
# @testset "gmm with gibbs" begin
#     w = [0.5, 0.5]
#     μ = [-3.5, 0.5]
#     mixturemodel = Distributions.MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

#     N = 60
#     x = rand(mixturemodel, N)

#     gmm = GMM((; x=x))

#     samples = sample(
#         gmm,
#         Gibbs(
#             (
#                 z = IndependentMH(product_distribution([Categorical([0.3, 0.7]) for _ in 1:60])),
#                 w = IndependentMH(Dirichlet(2, 1.0)),
#                 μ = RandomWalkMH(1),
#             ),
#         ),
#         100000;
#         initial_params=(z=rand(Categorical([0.3, 0.7]), 60), μ=[-3.5, 0.5], w=[0.3, 0.7]),
#     )

#     z_samples = [sample.values.z for sample in samples][20001:end]
#     μ_samples = [sample.values.μ for sample in samples][20001:end]
#     w_samples = [sample.values.w for sample in samples][20001:end]

#     # thin these samples
#     z_samples = z_samples[1:100:end]
#     μ_samples = μ_samples[1:100:end]
#     w_samples = w_samples[1:100:end]

#     mean(μ_samples)
#     mean(w_samples)
# end
