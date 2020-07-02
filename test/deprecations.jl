@testset "deprecations.jl" begin
    @test_deprecated AbstractMCMC.transitions(MySample(1, 2.0), MyModel(), MySampler())
    @test_deprecated AbstractMCMC.transitions(MySample(1, 2.0), MyModel(), MySampler(), 3)
end