# State Space Models

Patterns for state space models in RxInfer.jl, with real syntax from working examples.

## Linear Gaussian SSM (Kalman Filter)

### Rotating State Space Model

```julia
# From Kalman filtering and smoothing example
@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

# Setup matrices
θ = π / 35
A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]  # Rotation
B = diageye(2)                           # Observation
Q = 25.0 * diageye(2)                    # Observation noise
P = diageye(2)                           # Process noise

# Initial state prior
x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))

# Inference
result = infer(
    model = rotate_ssm(x0=x0, A=A, B=B, P=P, Q=Q), 
    data = (y = y,),
    free_energy = true
)

# Access results
xmarginals = result.posteriors[:x]
logevidence = -result.free_energy
```

### Plotting with Uncertainty

```julia
# Plot estimated signal with confidence ribbon
plot!(getindex.(mean.(xmarginals), 1), 
      ribbon = getindex.(var.(xmarginals), 1) .|> sqrt, 
      fillalpha = 0.5, 
      label = "Estimated Signal (dim-1)")
```

## Hidden Markov Model

```julia
# From HMM example
@model function hidden_markov_model(x)
    # Transition and emission matrices as Dirichlet
    A ~ DirichletCollection(ones(3,3))
    B ~ DirichletCollection([ 10.0 1.0 1.0; 
                              1.0 10.0 1.0; 
                              1.0 1.0 10.0 ])
    
    # Initial state
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    s_prev = s_0
    
    # Dynamics
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A) 
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end

# Constraints for VMP
@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end

# Initialization
imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(B) = vague(DirichletCollection, (3, 3)) 
    q(s) = vague(Categorical, 3)
end

# Inference
result = infer(
    model         = hidden_markov_model(), 
    data          = (x = x_data,),
    constraints   = hidden_markov_model_constraints(),
    initialization = imarginals, 
    returnvars    = (A = KeepLast(), B = KeepLast(), s = KeepLast()), 
    iterations    = 20, 
    free_energy   = true
)

# Access transition matrix
mean(result.posteriors[:A])

# Get most likely state sequence
argmax.(ReactiveMP.probvec.(result.posteriors[:s]))
```

## Signal Identification (Two Hidden Signals)

```julia
# From Kalman filtering - identifying two latent signals
@model function identification_problem(f, y, m_x_0, τ_x_0, a_x, b_x, m_w_0, τ_w_0, a_w, b_w, a_y, b_y)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x_i_min = x0
    w_i_min = w0

    local x, w, s
    
    for i in 1:length(y)
        x[i] ~ Normal(mean = x_i_min, precision = τ_x)
        w[i] ~ Normal(mean = w_i_min, precision = τ_w)
        s[i] := f(x[i], w[i])  # Deterministic combination
        y[i] ~ Normal(mean = s[i], precision = τ_y)
        
        x_i_min = x[i]
        w_i_min = w[i]
    end
end

constraints = @constraints begin 
    q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
end
```

## Smoothing with Missing Data

```julia
# From Kalman filtering example
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

# Handle missing data
missing_data = similar(noisy_data, Union{Float64, Missing})
copyto!(missing_data, noisy_data)
for index in 100:125
    missing_data[index] = missing
end

# Inference fills in missing values
result = infer(
    model = smoothing(x0=x0_prior), 
    data  = (y = missing_data,), 
    constraints = constraints,
    initialization = initm, 
    returnvars = (x = KeepLast(),),
    iterations = 20
)
```

## Streaming State Estimation

```julia
# From Kalman example - @autoupdates for online inference
autoupdates = @autoupdates begin 
    m_x_0, τ_x_0 = mean_precision(q(x))
    m_w_0, τ_w_0 = mean_precision(q(w))
    a_x = shape(q(τ_x)) 
    b_x = rate(q(τ_x))
end

engine = infer(
    model         = rx_identification(f=smooth_min),
    constraints   = rx_constraints,
    data          = (y = rx_real_y,),
    autoupdates   = autoupdates,
    keephistory   = 1000,
    historyvars   = KeepLast(),
    autostart     = true,
)

# Access streaming history
rx_xmarginals = engine.history[:x]
```

## Nonlinear SSM with Linearization

```julia
# Custom nonlinear function
function smooth_min(x, y)    
    if x < y
        return x + 1e-4 * y
    else
        return y + 1e-4 * x
    end
end

# Use @meta for approximation method
min_meta = @meta begin 
    smooth_min() -> Linearization()
end

result = infer(
    model = identification_problem(f=smooth_min, ...),
    meta = min_meta,
    ...
)
```

## Related Examples

- `scripts/Basic Examples/Kalman filtering and smoothing/`
- `scripts/Basic Examples/Hidden Markov Model/`
- `scripts/Basic Examples/Forgetting Factors for Online Inference/`
- `scripts/Advanced Examples/Bayesian Structured Time Series/`
