"""
DataGeneration - Synthetic data generation for multinomial models.

Provides configurable data generators for classification and regression.
"""
module DataGeneration

using Random
using Distributions
using LinearAlgebra

export generate_multinomial_data, generate_regression_data
export DataParams, ClassificationParams, RegressionParams

# Import softmax from SimplexUtils (will be included after)
include("SimplexUtils.jl")
using .SimplexUtils: softmax, logistic_stick_breaking

"""
    struct ClassificationParams

Parameters for multinomial classification data generation.
"""
struct ClassificationParams
    N::Int              # Trials per observation
    k::Int              # Number of categories
    nsamples::Int       # Number of samples
    seed::Int           # Random seed
end

function ClassificationParams(config::Dict)
    cls = config["classification"]
    ClassificationParams(
        cls["N"],
        cls["k"],
        cls["nsamples"],
        config["general"]["seed"]
    )
end

"""
    struct RegressionParams

Parameters for multinomial regression data generation.
"""
struct RegressionParams
    N::Int              # Trials per observation
    k::Int              # Feature dimension
    nsamples::Int       # Number of samples
    seed::Int           # Random seed
    transform::Function # Feature transform
end

function RegressionParams(config::Dict)
    reg = config["regression"]
    transform_name = get(reg, "feature_transform", "identity")
    transform = if transform_name == "sin"
        sin
    elseif transform_name == "tanh"
        tanh
    else
        identity
    end
    
    RegressionParams(
        reg["N"],
        reg["k"],
        reg["nsamples"],
        config["general"]["seed"],
        transform
    )
end

"""
    generate_multinomial_data(params::ClassificationParams) -> NamedTuple

Generate multinomial classification data.

Returns:
- X: Vector of observation vectors
- Ψ: True natural parameters
- p: True probabilities
"""
function generate_multinomial_data(params::ClassificationParams)
    rng = MersenneTwister(params.seed)
    
    # Generate true parameters
    Ψ = randn(rng, params.k)
    p = softmax(Ψ)
    
    # Generate observations
    X = rand(rng, Multinomial(params.N, p), params.nsamples)
    X = [X[:, i] for i in 1:size(X, 2)]
    
    return (X=X, Ψ=Ψ, p=p, params=params)
end

# Convenience method using config
function generate_multinomial_data(config::Dict)
    params = ClassificationParams(config)
    return generate_multinomial_data(params)
end

"""
    generate_regression_data(params::RegressionParams) -> NamedTuple

Generate multinomial regression data with covariates.

Returns:
- obs: Vector of observation vectors
- X: Vector of feature matrices
- β: True regression coefficients
- p: True probabilities for each observation
"""
function generate_regression_data(params::RegressionParams)
    rng = MersenneTwister(params.seed)
    
    # Generate true coefficients
    β = randn(rng, params.k)
    
    # Generate feature matrices
    X_raw = randn(rng, params.nsamples, params.k, params.k)
    X = [X_raw[i, :, :] for i in 1:size(X_raw, 1)]
    
    # Apply feature transform
    Ψ = params.transform.(X)
    
    # Compute probabilities and sample
    p = map(x -> logistic_stick_breaking(x * β), Ψ)
    obs = map(prob -> rand(rng, Multinomial(params.N, prob)), p)
    
    return (obs=obs, X=X, β=β, p=p, Ψ=Ψ, params=params)
end

# Convenience method using config
function generate_regression_data(config::Dict)
    params = RegressionParams(config)
    return generate_regression_data(params)
end

export ClassificationParams, RegressionParams

end # module
