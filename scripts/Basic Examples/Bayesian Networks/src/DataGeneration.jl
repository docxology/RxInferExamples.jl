"""
DataGeneration - Configurable synthetic data generation for the Sprinkler model.

Generates samples from the Bayesian Network model with configurable CPTs.
"""
module DataGeneration

using Random

export generate_samples, to_onehot, prepare_learning_data
export get_model_params, sample_statistics

"""
    struct ModelParams

Container for model parameters from configuration.
"""
struct ModelParams
    p_cloudy::Float64
    p_rain_given_cloudy::Vector{Float64}
    p_sprinkler_given_cloudy::Vector{Float64}
    p_wet_given_sprinkler_rain::Matrix{Float64}
end

"""
    get_model_params(config::Dict) -> ModelParams

Extract model parameters from configuration dictionary.
"""
function get_model_params(config::Dict)
    model = config["model"]
    
    # Convert nested array to matrix
    p_wet = model["p_wet_given_sprinkler_rain"]
    p_wet_matrix = [p_wet[1][1] p_wet[1][2]; p_wet[2][1] p_wet[2][2]]
    
    return ModelParams(
        model["p_cloudy"],
        model["p_rain_given_cloudy"],
        model["p_sprinkler_given_cloudy"],
        p_wet_matrix
    )
end

"""
    generate_samples(n::Int, params::ModelParams; seed::Int=42) -> NamedTuple

Generate n samples from the Sprinkler model using specified parameters.

Returns a named tuple with:
- clouded: Vector of 1 (false) or 2 (true)
- rain: Vector of 1 (false) or 2 (true)  
- sprinkler: Vector of 1 (false) or 2 (true)
- wet_grass: Vector of 1 (false) or 2 (true)
"""
function generate_samples(n::Int, params::ModelParams; seed::Int=42)
    Random.seed!(seed)
    
    clouded = zeros(Int, n)
    rain = zeros(Int, n)
    sprinkler = zeros(Int, n)
    wet_grass = zeros(Int, n)
    
    for i in 1:n
        # Sample clouded (prior)
        clouded[i] = rand() < params.p_cloudy ? 2 : 1  # 2=true, 1=false
        
        # Sample rain given clouded
        rain_prob = params.p_rain_given_cloudy[clouded[i]]
        rain[i] = rand() < rain_prob ? 2 : 1
        
        # Sample sprinkler given clouded
        sprinkler_prob = params.p_sprinkler_given_cloudy[clouded[i]]
        sprinkler[i] = rand() < sprinkler_prob ? 2 : 1
        
        # Sample wet grass given sprinkler and rain
        wet_prob = params.p_wet_given_sprinkler_rain[sprinkler[i], rain[i]]
        wet_grass[i] = rand() < wet_prob ? 2 : 1
    end
    
    return (clouded=clouded, rain=rain, sprinkler=sprinkler, wet_grass=wet_grass)
end

# Convenience overload using default params
function generate_samples(n::Int; seed::Int=42, 
                         p_cloudy::Float64=0.5,
                         p_rain_given_cloudy::Vector{Float64}=[0.2, 0.8],
                         p_sprinkler_given_cloudy::Vector{Float64}=[0.5, 0.1])
    params = ModelParams(
        p_cloudy,
        p_rain_given_cloudy,
        p_sprinkler_given_cloudy,
        [0.0 0.9; 0.9 0.99]
    )
    return generate_samples(n, params, seed=seed)
end

"""
    to_onehot(samples::Vector{Int}, n_categories::Int=2) -> Vector{Vector{Float64}}

Convert integer samples to one-hot encoded vectors.
"""
function to_onehot(samples::Vector{Int}, n_categories::Int=2)
    return [[i == s ? 1.0 : 0.0 for i in 1:n_categories] for s in samples]
end

"""
    prepare_learning_data(n::Int, params::ModelParams; seed::Int=42) -> NamedTuple

Generate and prepare one-hot encoded data for learning.
"""
function prepare_learning_data(n::Int, params::ModelParams; seed::Int=42)
    samples = generate_samples(n, params, seed=seed)
    
    return (
        clouded_data = to_onehot(samples.clouded),
        rain_data = to_onehot(samples.rain),
        sprinkler_data = to_onehot(samples.sprinkler),
        wet_grass_data = to_onehot(samples.wet_grass)
    )
end

# Convenience overload
function prepare_learning_data(n::Int; seed::Int=42)
    params = ModelParams(0.5, [0.2, 0.8], [0.5, 0.1], [0.0 0.9; 0.9 0.99])
    return prepare_learning_data(n, params, seed=seed)
end

"""
    sample_statistics(samples::NamedTuple) -> Dict

Compute empirical statistics from samples.
"""
function sample_statistics(samples::NamedTuple)
    n = length(samples.clouded)
    
    return Dict(
        "n_samples" => n,
        "p_cloudy_empirical" => count(==(2), samples.clouded) / n,
        "p_rain_empirical" => count(==(2), samples.rain) / n,
        "p_sprinkler_empirical" => count(==(2), samples.sprinkler) / n,
        "p_wet_grass_empirical" => count(==(2), samples.wet_grass) / n
    )
end

export ModelParams

end # module
