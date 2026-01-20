# Core Concepts

Foundational concepts for understanding RxInfer.jl, with real syntax from working examples.

## Factor Graphs

A **factor graph** is a bipartite graph representation of probabilistic models:

```
Variables (circles)  ←→  Factors (squares)
    x₁                      f₁(x₁, x₂)
    x₂                      f₂(x₂, x₃)
    x₃                      ...
```

## Message Passing

RxInfer uses **reactive message passing** for efficient inference:

```julia
# Messages propagate beliefs through the graph
# Forward messages: μ(x) from prior to likelihood
# Backward messages: m(x) from data to parameters
```

### Message Types

| Algorithm | Purpose | When to Use |
|-----------|---------|-------------|
| **Sum-Product** | Exact marginals | Conjugate models |
| **VMP** | Mean-field variational | Large models |
| **EP** | Moment-matching | Non-conjugate likelihood |

## The `infer` Function

Core inference function with all options:

```julia
# From Kalman filter example
result = infer(
    model          = rotate_ssm(x0=x0, A=A, B=B, P=P, Q=Q), 
    data           = (y = y,),
    free_energy    = true
)

# From HMM example - full options
result = infer(
    model          = hidden_markov_model(), 
    data           = (x = x_data,),
    constraints    = hidden_markov_model_constraints(),
    initialization = imarginals, 
    returnvars     = ireturnvars, 
    iterations     = 20, 
    free_energy    = true
)
```

### Inference Options Reference

| Option | Type | Description |
|--------|------|-------------|
| `model` | Model | From `@model` macro |
| `data` | NamedTuple | Observed variables |
| `constraints` | Constraints | From `@constraints` macro |
| `initialization` | Init | From `@initialization` macro |
| `meta` | Meta | From `@meta` macro |
| `autoupdates` | AutoUpdates | From `@autoupdates` macro |
| `iterations` | Int | VMP iterations (default: 1) |
| `free_energy` | Bool | Compute Bethe Free Energy |
| `returnvars` | Tuple | Which posteriors to return |
| `options` | NamedTuple | Advanced options |

## Bethe Free Energy

**Bethe Free Energy (BFE)** quantifies inference quality:

```julia
# From Kalman example
result = infer(
    model = rotate_ssm(x0=x0, A=A, B=B, P=P, Q=Q), 
    data = (y = y,),
    free_energy = true
)

logevidence = -result.free_energy  # Negative log evidence

# Plot convergence (from Linear Regression)
plot(result.free_energy, 
     title="Free energy", 
     xlabel="Iteration", 
     ylabel="Free energy [nats]")
```

### Interpretation

- **Decreasing F**: Converging to better approximation
- **Stable F**: Reached local optimum
- For exact inference (conjugate models): `F = -log p(y)`

## Conjugate Pairs

RxInfer exploits **conjugate relationships** for closed-form updates:

| Likelihood | Prior | RxInfer Syntax |
|------------|-------|----------------|
| `Bernoulli(θ)` | `Beta(a, b)` | `θ ~ Beta(a, b); y ~ Bernoulli(θ)` |
| `Normal(μ, σ)` | `Normal(μ₀, σ₀)` | `μ ~ Normal(mean=0, var=1); y ~ Normal(mean=μ, var=σ)` |
| `Normal(μ, τ)` | `Gamma(α, β)` | `τ ~ Gamma(shape=a, rate=b); y ~ Normal(mean=μ, precision=τ)` |
| `Categorical(π)` | `Dirichlet(α)` | `π ~ Dirichlet(ones(K)); z ~ Categorical(π)` |
| `MvNormal(μ, Σ)` | `InverseWishart(ν, S)` | `Σ ~ InverseWishart(ν, S); y ~ MvNormal(mean=μ, covariance=Σ)` |

## Accessing Results

```julia
# Single posterior
θestimated = result.posteriors[:θ]

# Multiple posteriors
xmarginals = result.posteriors[:x]  # Vector for indexed variables

# Statistics
mean(θestimated)
var(θestimated)
std(θestimated)

# For multivariate
mean.(xmarginals)  # Vector of means
var.(xmarginals)   # Vector of variances

# Plotting with ribbon
plot!(getindex.(mean.(xmarginals), 1), 
      ribbon = getindex.(var.(xmarginals), 1) .|> sqrt,
      label = "Estimated Signal")
```

## Related Examples

- `scripts/Basic Examples/Coin Toss Model/` - Beta-Bernoulli conjugacy
- `scripts/Basic Examples/Kalman filtering and smoothing/` - Full SSM inference
- `scripts/Basic Examples/Hidden Markov Model/` - Discrete state inference
