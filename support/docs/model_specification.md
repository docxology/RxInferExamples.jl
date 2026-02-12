# Model Specification

Guide to specifying probabilistic models in RxInfer using the `@model` macro, with real syntax from working examples.

## The @model Macro

Models are defined using the `@model` macro. The macro transforms Julia code into a factor graph representation.

```julia
# From Coin Toss example
@model function coin_model(y, a, b)
    θ ~ Beta(a, b)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

# Usage
result = infer(
    model = coin_model(a = 4.0, b = 8.0), 
    data  = (y = dataset,)
)
```

## Syntax Elements

### Variable Declaration

| Syntax | Meaning | Example |
|--------|---------|---------|
| `x ~ Distribution()` | Random variable | `θ ~ Beta(a, b)` |
| `y[i] ~ ...` | Indexed variables | `y[i] ~ Bernoulli(θ)` |
| `x := f(...)` | Deterministic transformation | `s[i] := f(x[i], w[i])` |
| `y .~ Distribution()` | Vectorized/Broadcasted | `y .~ Normal(mean = a .* x .+ b, variance = 1.0)` |

### Parameter Naming

```julia
# Named parameters (from Linear Regression)
@model function linear_regression(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)    
    y .~ Normal(mean = a .* x .+ b, variance = 1.0)
end

# Precision vs Variance parameterization
τ ~ Gamma(shape = a_x, rate = b_x)
y ~ Normal(mean = μ, precision = τ)  # precision = 1/variance
```

## Advanced Patterns

### State Space Models (SSM)

#### Linear Gaussian SSM (Kalman Filter)

```julia
# From Kalman filtering example
@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end
```

#### Hidden Markov Model (HMM)

```julia
# From HMM example
@model function hidden_markov_model(x)
    A ~ DirichletCollection(ones(3,3))
    B ~ DirichletCollection([ 10.0 1.0 1.0; 
                              1.0 10.0 1.0; 
                              1.0 1.0 10.0 ])
    
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    s_prev = s_0
    
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A) 
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
end
```

### Hierarchical Models

```julia
# From Bayesian Linear Regression (partially pooled)
@model function partially_pooled(patient_codes, weeks, data)
    μ_α ~ Normal(mean = 0.0, var = 250000.0)
    σ_α ~ Gamma(shape = 1.75, scale = 45.54)
    
    n_patients = length(unique(patient_codes))
    local α
    
    for i in 1:n_patients
        α[i] ~ Normal(mean = μ_α, precision = σ_α)
    end
    
    for i in 1:length(patient_codes)
        # Using hierarchical parameter α[patient_codes[i]]
        y[i] ~ Normal(mean = α[patient_codes[i]] + β * weeks[i], precision = σ)
    end
end
```

## Meta Information (@meta)

The `@meta` macro is crucial for handling nonlinear node approximations and custom rules.

### defining Approximations

For nonlinear deterministic nodes (like `y := f(x)`), exact inference is often impossible. You must specify an approximation method (e.g., Linearization, Unscented).

```julia
# From Kalman example with nonlinear transition
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
    # Or: smooth_min() -> Unscented(alpha=0.1, beta=2.0, kappa=0.0)
end

result = infer(
    model = identification_problem(f=smooth_min, ...),
    meta = min_meta,
    ...
)
```

## Constraints (@constraints)

Define factorization assumptions for variational inference (VMP).

```julia
# Mean-Field (everything factorized)
constraints = MeanField()

# Structured Factorization
@constraints function hidden_markov_model_constraints()
    # Factorize parameters from states, but keep states and initial state joint
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end
```

## Initialization (@initialization)

Set initial marginals (priors for the variational posterior) or messages.

```julia
# Marginal Initialization (standard)
imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(s) = vague(Categorical, 3)
end

# Message Initialization (for feedback loops/RNNs)
init = @initialization begin
    μ(x) = xinit  # Vector of initial messages
end
```

## Return Variables

Control which posteriors are returned to save memory.

```julia
# Keep last iteration only
ireturnvars = (
    A = KeepLast(),
    B = KeepLast(),
    s = KeepLast()
)

# Keep specific history
ireturnvars = (
    x = KeepEach(),  # Keep all iterations
)
```

## Debugging Models

1. **Check Graph Structure**: Use `RxInfer.GraphPPL.get_model_graph(model)` to inspect the generated graph.
2. **Dimension Mismatch**: Ensure `A`, `B`, and state vectors have compatible dimensions in `MvNormal` nodes.
3. **Undefined Variables**: Ensure all variables on the RHS of `~` are defined or passed as arguments.
4. **Constraints**: If inference fails or is terrible, check if your `@constraints` are too restrictive (e.g., breaking dependencies that *must* exist).

## Related Examples

- `scripts/Basic Examples/Coin Toss Model/`
- `scripts/Basic Examples/Hidden Markov Model/`
- `scripts/Basic Examples/Kalman filtering and smoothing/`
- `scripts/Basic Examples/Bayesian Linear Regression/`
