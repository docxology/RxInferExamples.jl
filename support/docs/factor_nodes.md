# Factor Nodes

Built-in and custom factor nodes in RxInfer.jl, with real syntax from working examples.

## Stochastic Nodes

### Continuous

| Node | Syntax | From Example |
|------|--------|--------------|
| Normal | `x ~ Normal(mean=μ, variance=σ²)` | Linear Regression |
| Normal (precision) | `x ~ Normal(mean=μ, precision=τ)` | Kalman |
| MvNormal | `x ~ MvNormalMeanCovariance(μ, Σ)` | Kalman |
| Gamma | `τ ~ Gamma(shape=α, rate=β)` | Linear Regression |
| Beta | `θ ~ Beta(a, b)` | Coin Toss |
| InverseGamma | `s ~ InverseGamma(α, β)` | Linear Regression |
| InverseWishart | `W ~ InverseWishart(ν, S)` | Multivariate Regression |

### Discrete

| Node | Syntax | From Example |
|------|--------|--------------|
| Bernoulli | `y ~ Bernoulli(θ)` | Coin Toss |
| Categorical | `s ~ Categorical(π)` | HMM |
| DiscreteTransition | `s[t] ~ DiscreteTransition(s_prev, A)` | HMM, POMDP |
| DirichletCollection | `A ~ DirichletCollection(M)` | HMM |

## Deterministic Nodes

```julia
# From Kalman identification example
s[i] := f(x[i], w[i])  # Deterministic transformation

# Arithmetic operations
y .~ Normal(mean = a .* x .+ b, variance = 1.0)

# From Hierarchical Regression
FVC_est[i] ~ α[patient_codes[i]] + β[patient_codes[i]] * weeks[i]
```

## DiscreteTransition Node

The `DiscreteTransition` node handles discrete state transitions:

```julia
# From HMM - 2D transition (state-to-state)
s[t] ~ DiscreteTransition(s_prev, A)   # A is transition matrix
x[t] ~ DiscreteTransition(s[t], B)     # B is emission matrix

# From POMDP - 3D transition (state × action → state)
current_state ~ DiscreteTransition(previous_state, B, previous_control)
```

## DirichletCollection Node

For learning transition matrices:

```julia
# From HMM
A ~ DirichletCollection(ones(3,3))  # Uniform prior
B ~ DirichletCollection([ 10.0 1.0 1.0; 
                          1.0 10.0 1.0; 
                          1.0 1.0 10.0 ])  # Informative prior
```

## Vectorized Operations

```julia
# Dot-broadcast for multiple observations
y .~ Normal(mean = a .* x .+ b, variance = 1.0)

# With covariance matrix
y .~ MvNormal(mean = x .* a .+ b, covariance = W)
```

## Meta for Nonlinear Nodes

```julia
# From Kalman example - custom nonlinear function
function smooth_min(x, y)    
    if x < y
        return x + 1e-4 * y
    else
        return y + 1e-4 * x
    end
end

# Specify approximation method
min_meta = @meta begin 
    smooth_min() -> Linearization()
end

result = infer(
    model = model(f=smooth_min),
    meta = min_meta,
    ...
)
```

## Accessing Node Statistics

```julia
# From HMM - get probability vectors
ReactiveMP.probvec.(result.posteriors[:s])

# Most likely discrete state
argmax.(ReactiveMP.probvec.(result.posteriors[:s]))

# From Linear Regression
mean(result.posteriors[:A])  # Matrix mean
```

## Custom Message Rules

```julia
# From Linear Regression - custom call_rule
FVC_predicted = broadcast(results.posteriors[:FVC_est], Ref(results.posteriors[:σ])) do f, s
    return @call_rule NormalMeanPrecision(:out, Marginalisation) (m_μ = f, q_τ = s)
end
```

## Common Patterns

### Linear Transformation
```julia
y ~ Normal(mean = a * x + b, variance = σ²)
```

### Gaussian Process Observation
```julia
x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
```

### Hierarchical Prior
```julia
μ_α ~ Normal(mean = 0.0, var = 250000.0)
σ_α ~ Gamma(shape = 1.75, scale = 45.54)
for i in 1:n_patients
    α[i] ~ Normal(mean = μ_α, precision = σ_α)
end
```

## Related Examples

- `scripts/Basic Examples/Coin Toss Model/` - Beta, Bernoulli
- `scripts/Basic Examples/Hidden Markov Model/` - DiscreteTransition, DirichletCollection
- `scripts/Basic Examples/Kalman filtering and smoothing/` - MvNormal, Gamma
- `scripts/Basic Examples/Bayesian Linear Regression/` - Hierarchical structures
