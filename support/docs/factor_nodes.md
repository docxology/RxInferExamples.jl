# Factor Nodes

Built-in and custom factor nodes in RxInfer.jl, with real syntax from working examples.

## Stochastic Nodes

RxInfer supports a wide range of distributions.

### Univariate Continuous

| Node | Canonical Parameters | Syntax | Usage Pattern |
|------|----------------------|--------|---------------|
| **Normal** | Mean, Variance | `Normal(mean=μ, variance=σ²)` | General noise, priors |
| **Normal** | Mean, Precision | `Normal(mean=μ, precision=τ)` | Conjugate to Gamma precision |
| **Gamma** | Shape, Rate | `Gamma(shape=α, rate=β)` | Precision priors |
| **Beta** | Alpha, Beta | `Beta(a, b)` | Probability priors (Bernoulli) |
| **InverseGamma** | Shape, Scale | `InverseGamma(shape=α, scale=β)` | Variance priors |

### Multivariate Continuous

| Node | Canonical Parameters | Syntax | Usage Pattern |
|------|----------------------|--------|---------------|
| **MvNormal** | Mean, Covariance | `MvNormalMeanCovariance(μ, Σ)` | Kalman State Transition |
| **MvNormal** | Mean, Precision | `MvNormalMeanPrecision(μ, Λ)` | Conjugate to Wishart |
| **Dirichlet** | Alpha (vector) | `Dirichlet(α)` | Categorical priors |
| **InverseWishart** | D.o.F, Scale Matrix | `InverseWishart(ν, S)` | Covariance matrix priors |

### Discrete

| Node | Canonical Parameters | Syntax | Usage Pattern |
|------|----------------------|--------|---------------|
| **Bernoulli** | p | `Bernoulli(θ)` | Binary outcomes |
| **Categorical** | p (vector) | `Categorical(p)` | State variables in HMM |
| **DiscreteTransition** | x_prev, Transition Matrix | `DiscreteTransition(x_prev, A)` | HMM transitions |
| **Binomial** | n, p | `Binomial(n, p)` | Count data |

## Specialized Nodes

### DiscreteTransition

The `DiscreteTransition` node is essential for HMMs and POMDPs. It represents a state transition $P(x_t | x_{t-1})$.

```julia
# 2D Transition (Standard HMM)
# s[t] depends on s[t-1] and matrix A
s[t] ~ DiscreteTransition(s_prev, A)

# 3D Transition (Control/Action dependent)
# current_state depends on previous_state, Transition Tensor B, and action/control
current_state ~ DiscreteTransition(previous_state, B, previous_control)
```

### DirichletCollection

Used for learning transition matrices row-by-row. It creates a collection of independent Dirichlet distributions, one for each row of the matrix.

```julia
# Prior for a 3x3 transition matrix
# Initialize with uniform prior (ones)
A ~ DirichletCollection(ones(3,3))

# Initialize with strong diagonal preference (sticky HMM)
B ~ DirichletCollection([ 10.0 1.0 1.0; 
                          1.0 10.0 1.0; 
                          1.0 1.0 10.0 ])
```

## Deterministic Nodes

Deterministic nodes perform fixed calculations variables.

```julia
# Arithmetic
y .~ Normal(mean = a .* x .+ b, variance = 1.0)

# Custom Functions
s[i] := f(x[i], w[i])

# Logical/Switching
z[t] := switch_logic(modality[t], visual_input[t], auditory_input[t])
```

## Vectorized Operations

Dot-syntax (`.~`) vs Loop syntax.

```julia
# Loop (Explicit)
for i in 1:N
    y[i] ~ Normal(mean = x[i], variance = 1.0)
end

# Vectorized (Compact, often optimized)
y .~ Normal(mean = x, variance = 1.0)
```

## Meta for Nonlinear Nodes

When using deterministic nodes that are nonlinear (non-conjugate), you must specify how to approximate messages.

```julia
# Define function
function nonlinear_transition(x)
    return x^2 + sin(x)
end

# Define approximation
meta = @meta begin
    nonlinear_transition() -> Linearization() # Taylor expansion
end
```

Available approximations:

- `Linearization()`: First-order Taylor expansion around the mean (EKF-like).
- `Unscented(alpha=..., beta=..., kappa=...)`: Unscented Transform (UKF-like).
- `CVI()`: Conjugate Variational Inference (limited support).
