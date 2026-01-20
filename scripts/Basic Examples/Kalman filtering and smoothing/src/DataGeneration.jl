module DataGeneration

using Random
using Distributions
using LinearAlgebra
using StableRNGs
using RxInfer

export generate_rotating_ssm_data, generate_identification_data

"""
    generate_rotating_ssm_data(rng, A, B, P, Q, n)

Generate data for the rotating SSM example.
"""
function generate_rotating_ssm_data(rng, A, B, P, Q, n)
    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormalMeanCovariance(A * x_prev, P))
        y[i] = rand(rng, MvNormalMeanCovariance(B * x[i], Q))
        x_prev = x[i]
    end
    
    return x, y
end

"""
    generate_identification_data(f, n; seed = 123, x_i_min = -20.0, w_i_min = 20.0, noise = 20.0, real_x_τ = 0.1, real_w_τ = 1.0)

Generate data for the identification and smoothing examples.
"""
function generate_identification_data(f, n; seed = 123, x_i_min = -20.0, w_i_min = 20.0, noise = 20.0, real_x_τ = 0.1, real_w_τ = 1.0)
    rng = StableRNG(seed)

    real_x = Vector{Float64}(undef, n)
    real_w = Vector{Float64}(undef, n)
    real_y = Vector{Float64}(undef, n)

    for i in 1:n
        real_x[i] = rand(rng, Normal(x_i_min, sqrt(1.0 / real_x_τ)))
        real_w[i] = rand(rng, Normal(w_i_min, sqrt(1.0 / real_w_τ)))
        real_y[i] = rand(rng, Normal(f(real_x[i], real_w[i]), sqrt(noise)))

        x_i_min = real_x[i]
        w_i_min = real_w[i]
    end
    
    return real_x, real_w, real_y
end

end # module
