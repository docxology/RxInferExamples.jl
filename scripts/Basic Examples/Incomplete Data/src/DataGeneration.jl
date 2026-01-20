module DataGeneration

using Random
using LinearAlgebra
using Distributions

export generate_incomplete_data

"""
    generate_incomplete_data(rng, n_samples, dimension)

Generate synthetic data for the Incomplete Data problem.
Returns a NamedTuple with (mean, covariance, complete_data, observed_data).
"""
function generate_incomplete_data(rng, n_samples, dimension)
    # Generate random mean and covariance
    # Using fixed values from original script for reproducibility if needed, but rng allows variation
    # Original: real_m = [13.0, 1.0, 5.0, 4.0, -20.0, 10.0]
    # We will generate them randomly to make it more rigorous, or stick to a pattern if critical.
    # The original script used specific numbers, let's generate them but keep magnitude similar.
    
    real_m = randn(rng, dimension) .* 10.0 
    
    # Ensure positive definite covariance
    A = randn(rng, dimension, dimension)
    real_Λ = A' * A + I # Precision matrix essentially
    real_Σ = inv(real_Λ) # Covariance
    real_Σ = Hermitian(real_Σ)
    
    # Generate complete data
    dist = MvNormal(real_m, real_Σ)
    real_x = [rand(rng, dist) for _ in 1:n_samples]
    
    # Create incomplete data (introduce missing values)
    # Original: one missing value per sample at random position
    observed_data = Matrix{Union{Float64, Missing}}(undef, n_samples, dimension)
    
    for i in 1:n_samples
        sample = copy(real_x[i])
        # Introduce missing value
        missing_idx = rand(rng, 1:dimension)
        
        for j in 1:dimension
            if j == missing_idx
                observed_data[i, j] = missing
            else
                observed_data[i, j] = sample[j]
            end
        end
    end
    
    return (
        real_mean = real_m,
        real_precision = real_Λ,
        real_covariance = real_Σ,
        complete_data = real_x,
        observed_data = observed_data
    )
end

end # module
