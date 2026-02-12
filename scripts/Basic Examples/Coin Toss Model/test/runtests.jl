using Test
using RxInfer
using Random
using Distributions

# Include the main script logic (but avoid running main())
# We need to expose functions for testing.
# Since the script is not a module, we include it in a module to inspect.
module CoinTossModelTest
    const PROGRAM_FILE = "" # Prevent main() from running
    include("../Coin Toss Model.jl")
end

using .CoinTossModelTest

@testset "Coin Toss Model Tests" begin
    
    @testset "Data Generation" begin
        rng = Random.MersenneTwister(42)
        n = 100
        theta = 0.8
        data = CoinTossModelTest.generate_data(rng, n, theta)
        
        @test length(data) == n
        @test all(x -> x in [0.0, 1.0], data)
        @test 0.7 <= mean(data) <= 0.9 # Rough check
    end

    @testset "Batch Inference" begin
        data = [1.0, 1.0, 1.0, 0.0]
        prior_a = 1.0
        prior_b = 1.0
        
        result = CoinTossModelTest.run_batch_inference(data, prior_a, prior_b)
        posterior = result.posteriors[:θ]
        
        # Exact posterior check: Beta(1+3, 1+1) = Beta(4, 2)
        @test mean(posterior) ≈ 4/6
    end

    @testset "Online Inference" begin
        data = [1.0, 0.0]
        prior_a = 1.0
        prior_b = 1.0
        
        posteriors = CoinTossModelTest.run_online_inference(data, prior_a, prior_b)
        
        @test length(posteriors) == 2
        @test mean(posteriors[1]) ≈ 2/3 # Beta(2, 1) after first head
        @test mean(posteriors[2]) ≈ 2/4 # Beta(2, 2) after second tail
    end

    @testset "Validation Logic" begin
        # Perfect posterior
        posterior = Beta(100, 100) # Mean 0.5
        real_theta = 0.5
        threshold = 0.1
        
        val_result = CoinTossModelTest.validate_results(posterior, real_theta, threshold)
        @test val_result.passed == true
        
        # Bad posterior
        real_theta = 0.9
        val_result = CoinTossModelTest.validate_results(posterior, real_theta, threshold)
        @test val_result.passed == false
    end
end
