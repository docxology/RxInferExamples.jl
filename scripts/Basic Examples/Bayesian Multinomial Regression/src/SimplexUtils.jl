"""
SimplexUtils - Utilities for simplex coordinate transformations.

Provides functions for logistic stick-breaking and related transforms.
"""
module SimplexUtils

export σ, σ_inv, softmax, logistic_stick_breaking
export ψ_to_π, π_to_ψ, jacobian_det
export compute_simplex_density

"""Sigmoid function."""
σ(x) = 1 / (1 + exp(-x))

"""Inverse sigmoid (logit)."""
σ_inv(x) = log(x / (1 - x))

"""
    softmax(x::AbstractVector) -> Vector

Compute softmax of a vector.
"""
function softmax(x::AbstractVector)
    ex = exp.(x .- maximum(x))  # Numerical stability
    return ex ./ sum(ex)
end

"""
    logistic_stick_breaking(ψ::AbstractVector) -> Vector

Convert natural parameters ψ to probability simplex via logistic stick-breaking.
"""
function logistic_stick_breaking(ψ::AbstractVector)
    K = length(ψ) + 1
    π = zeros(K)
    remaining = 1.0
    
    for k in 1:(K-1)
        π[k] = σ(ψ[k]) * remaining
        remaining -= π[k]
    end
    π[K] = remaining
    
    return π
end

"""
    ψ_to_π(ψ::Vector{Float64}) -> Vector

Convert ψ (natural parameters) to π (probabilities) using stick-breaking.
"""
function ψ_to_π(ψ::Vector{Float64})
    K = length(ψ) + 1
    π = zeros(K)
    for k in 1:(K-1)
        π[k] = σ(ψ[k]) * (1 - sum(π[1:(k-1)]))
    end
    π[K] = 1 - sum(π[1:(K-1)])
    return π
end

"""
    π_to_ψ(π::AbstractVector) -> Vector

Convert π (probabilities) to ψ (natural parameters).
"""
function π_to_ψ(π::AbstractVector)
    K = length(π)
    ψ = zeros(K-1)
    ψ[1] = σ_inv(π[1])
    for k in 2:(K-1)
        ψ[k] = σ_inv(π[k] / (1 - sum(π[1:(k-1)])))
    end
    return ψ
end

"""
    jacobian_det(π::AbstractVector) -> Float64

Compute the Jacobian determinant for the stick-breaking transform.
"""
function jacobian_det(π::AbstractVector)
    K = length(π)
    det = 1.0
    for k in 1:(K-1)
        num = 1 - sum(π[1:(k-1)])
        den = π[k] * (1 - sum(π[1:k]))
        if den > 0
            det *= num / den
        end
    end
    return det
end

"""
    compute_simplex_density(x, y, μ, Σ) -> Float64

Compute density in simplex coordinates (2-simplex).
"""
function compute_simplex_density(x::Float64, y::Float64, Σ::AbstractMatrix)
    # Check if point is inside triangle (2-simplex)
    if y < 0 || y > 1 || x < 0 || x > 1 || (x + y) > 1
        return 0.0
    end
    
    # Convert from simplex coordinates to π
    π1, π2, π3 = x, y, 1 - x - y
    
    try
        ψ = π_to_ψ([π1, π2, π3])
        # Compute Gaussian density
        d = length(ψ)
        detΣ = det(Σ)
        if detΣ <= 0
            return 0.0
        end
        normconst = 1 / sqrt((2π)^d * detΣ)
        exponent = -0.5 * ψ' * inv(Σ) * ψ
        return normconst * exp(exponent) * abs(jacobian_det([π1, π2, π3]))
    catch
        return 0.0
    end
end

using LinearAlgebra: det, inv

end # module
