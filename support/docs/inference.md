# Inference

Running Bayesian inference with RxInfer.jl, with real syntax from working examples.

## Basic Inference

```julia
# From Coin Toss example
using RxInfer, Random

@model function coin_model(y, a, b)
    θ ~ Beta(a, b)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

result = infer(
    model = coin_model(a = 4.0, b = 8.0), 
    data  = (y = dataset,)
)

θestimated = result.posteriors[:θ]
```

## Full Inference Example

```julia
# From HMM example - complete inference call
result = infer(
    model         = hidden_markov_model(), 
    data          = (x = x_data,),
    constraints   = hidden_markov_model_constraints(),
    initialization = imarginals, 
    returnvars    = ireturnvars, 
    iterations    = 20, 
    free_energy   = true
)
```

## Inference Parameters Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model` | Model instance from @model | `coin_model(a=4.0, b=8.0)` |
| `data` | Named tuple of observations | `(y = dataset,)` |
| `constraints` | Factorization constraints | `MeanField()` |
| `initialization` | Initial marginals/messages | `@initialization begin ... end` |
| `meta` | Approximation methods | `@meta begin ... end` |
| `iterations` | VMP iterations | `20` |
| `free_energy` | Compute BFE | `true` |
| `returnvars` | Which posteriors | `(a = KeepLast(),)` |
| `options` | Advanced options | `(limit_stack_depth = 500,)` |

## Constraints for Variational Inference

### Mean-Field Shorthand

```julia
# From Linear Regression with unknown noise
results = infer(
    model       = linear_regression_unknown_noise(), 
    data        = (y = y_data, x = x_data), 
    constraints = MeanField(),
    iterations  = 20,
    free_energy = true
)
```

### Custom Factorization

```julia
# From HMM example
@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end

# From Linear Regression hierarchical
@constraints function partially_pooled_constraints()
    q(μ_α, σ_α, μ_β, σ_β, σ) = q(μ_α)q(σ_α)q(μ_β)q(σ_β)q(σ)
    q(μ_α, σ_α, α) = q(μ_α, α)q(σ_α)
    q(μ_β, σ_β, β) = q(μ_β, β)q(σ_β)
    q(FVC_est, σ) = q(FVC_est)q(σ) 
end
```

## Initialization

### Marginal Initialization

```julia
# From HMM example
imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(B) = vague(DirichletCollection, (3, 3)) 
    q(s) = vague(Categorical, 3)
end
```

### Message Initialization

```julia
# From Kalman filtering example
init = @initialization begin
    μ(x) = xinit  # Vector of messages
    μ(w) = winit
    q(τ_x) = GammaShapeRate(a_x, b_x)
    q(τ_w) = GammaShapeRate(a_w, b_w)
    q(τ_y) = GammaShapeRate(a_y, b_y)
end
```

### Inline Initialization

```julia
# From Linear Regression
results = infer(
    model = linear_regression(), 
    data  = (y = y_data, x = x_data), 
    initialization = @initialization(μ(b) = NormalMeanVariance(0.0, 100.0))
)
```

## Streaming/Reactive Inference

```julia
# From Kalman filtering example - @autoupdates for streaming
autoupdates = @autoupdates begin 
    m_x_0, τ_x_0 = mean_precision(q(x))
    m_w_0, τ_w_0 = mean_precision(q(w))
    a_x = shape(q(τ_x)) 
    b_x = rate(q(τ_x))
    a_y = shape(q(τ_y))
    b_y = rate(q(τ_y))
    a_w = shape(q(τ_w)) 
    b_w = rate(q(τ_w))
end

engine = infer(
    model         = rx_identification(f=smooth_min),
    constraints   = rx_constraints,
    data          = (y = rx_real_y,),
    autoupdates   = autoupdates,
    meta          = rx_meta,
    returnvars    = (:x, :w, :τ_x, :τ_w, :τ_y, :s),
    keephistory   = 1000,
    historyvars   = KeepLast(),
    initialization = init,
    iterations    = 10,
    free_energy   = true, 
    autostart     = true,
)

# Access history
rx_smarginals = engine.history[:s]
rx_xmarginals = engine.history[:x]
```

## Free Energy Convergence

```julia
# From Linear Regression
plot(results.free_energy, 
     title="Free energy", 
     xlabel="Iteration", 
     ylabel="Free energy [nats]", 
     legend=false)

# Skip first iteration (initialization effect)
plot(results.free_energy[2:end], ...)
```

## Accessing Results

### Posteriors

```julia
# Single variable
θestimated = result.posteriors[:θ]

# Indexed variables
xmarginals = result.posteriors[:x]  # Vector

# Statistics
mean(θestimated)
var(θestimated)
std(θestimated)

# For matrix distributions
mean(result.posteriors[:A])  # Returns matrix
```

### Multiple Iterations

```julia
# When using iterations > 1 without KeepLast()
τ_x_marginals = result.posteriors[:τ_x]  # Vector per iteration
τ_x_marginals[end]  # Final iteration
```

## Advanced Options

```julia
# Large models need stack limit
result = infer(
    model = large_model(),
    data = data,
    options = (limit_stack_depth = 500,),
    ...
)
```

## Related Examples

- `scripts/Basic Examples/Coin Toss Model/` - Basic inference
- `scripts/Basic Examples/Hidden Markov Model/` - Full VMP with constraints  
- `scripts/Basic Examples/Kalman filtering and smoothing/` - Streaming inference
- `scripts/Basic Examples/Bayesian Linear Regression/` - Hierarchical models
