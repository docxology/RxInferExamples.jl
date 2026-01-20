module Models

using RxInfer
using Distributions
using LinearAlgebra

export rotate_ssm, identification_problem, rx_identification, smoothing, smooth_min

"""
    smooth_min(x, y)

Smoothed version of `min` without zero-ed derivatives.
"""
function smooth_min(x, y)    
    if x < y
        return x + 1e-4 * y
    else
        return y + 1e-4 * x
    end
end

@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

@model function identification_problem(f, y, m_x_0, τ_x_0, a_x, b_x, m_w_0, τ_w_0, a_w, b_w, a_y, b_y, λ)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x_i_min = x0
    w_i_min = w0

    local x
    local w
    local s
    
    for i in 1:length(y)
        # Random walk with damping (AR(1))
        # λ < 1.0 stabilizes the variance
        x[i] ~ Normal(mean = λ * x_i_min, precision = τ_x)
        w[i] ~ Normal(mean = λ * w_i_min, precision = τ_w)
        s[i] := f(x[i], w[i])
        y[i] ~ Normal(mean = s[i], precision = τ_y)
        
        x_i_min = x[i]
        w_i_min = w[i]
    end
end

@model function rx_identification(f, m_x_0, τ_x_0, m_w_0, τ_w_0, a_x, b_x, a_y, b_y, a_w, b_w, y, λ)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x ~ Normal(mean = λ * x0, precision = τ_x)
    w ~ Normal(mean = λ * w0, precision = τ_w)

    s := f(x, w)
    y ~ Normal(mean = s, precision = τ_y)
end

@model function smoothing(x0, y)
    P ~ Gamma(shape = 0.001, scale = 0.001)
    x_prior ~ Normal(mean = mean(x0), var = var(x0)) 

    local x
    x_prev = x_prior

    for i in 1:length(y)
        x[i] ~ Normal(mean = x_prev, precision = 1.0)
        y[i] ~ Normal(mean = x[i], precision = P)
        
        x_prev = x[i]
    end
end

end # module
