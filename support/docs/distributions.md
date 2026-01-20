# Probability Distributions

Reference for distributions supported in RxInfer.jl, with real syntax from working examples.

## Univariate Continuous

| Distribution | Syntax | Parameters |
|--------------|--------|------------|
| Normal (μ,σ²) | `Normal(mean = μ, variance = σ²)` | mean, variance |
| Normal (μ,τ) | `Normal(mean = μ, precision = τ)` | mean, precision |
| Gamma | `Gamma(shape = α, rate = β)` | shape, rate |
| Gamma | `Gamma(shape = α, scale = θ)` | shape, scale |
| Beta | `Beta(a, b)` | shape parameters |
| InverseGamma | `InverseGamma(α, β)` | shape, scale |

### Real Examples

```julia
# From Coin Toss
θ ~ Beta(a, b)

# From Linear Regression
a ~ Normal(mean = 0.0, variance = 1.0)
b ~ Normal(mean = 0.0, variance = 100.0)
s ~ InverseGamma(1.0, 1.0)

# From Kalman identification
τ_x ~ Gamma(shape = a_x, rate = b_x)
y ~ Normal(mean = s, precision = τ_y)
```

## Multivariate Continuous

| Distribution | Syntax | Parameters |
|--------------|--------|------------|
| MvNormal | `MvNormalMeanCovariance(μ, Σ)` | mean, covariance |
| MvNormal | `MvNormalMeanPrecision(μ, Λ)` | mean, precision |
| InverseWishart | `InverseWishart(ν, S)` | degrees of freedom, scale |

### Real Examples

```julia
# From Kalman filter
x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
y[i] ~ MvNormalMeanCovariance(B * x[i], Q)

# From Multivariate Linear Regression
a ~ MvNormal(mean = zeros(dim), covariance = 100 * diageye(dim))
W ~ InverseWishart(dim + 2, 100 * diageye(dim))
y .~ MvNormal(mean = x .* a .+ b, covariance = W)
```

## Discrete Distributions

| Distribution | Syntax | Parameters |
|--------------|--------|------------|
| Bernoulli | `Bernoulli(p)` | success probability |
| Categorical | `Categorical(p)` | probability vector |

### Real Examples

```julia
# From Coin Toss
y[i] ~ Bernoulli(θ)

# From HMM
s_0 ~ Categorical(fill(1.0 / 3.0, 3))

# POMDP goal as one-hot
goal = Categorical(index_to_one_hot(grid_location_to_index((4, 3))))
```

## Matrix Distributions

| Distribution | Syntax | Parameters |
|--------------|--------|------------|
| Dirichlet | `Dirichlet(α)` | concentration vector |
| DirichletCollection | `DirichletCollection(A)` | concentration matrix |

### Real Examples

```julia
# From HMM - transition and emission matrices
A ~ DirichletCollection(ones(3,3))
B ~ DirichletCollection([ 10.0 1.0 1.0; 
                          1.0 10.0 1.0; 
                          1.0 1.0 10.0 ])
```

## Transition Distributions

```julia
# From HMM
s[t] ~ DiscreteTransition(s_prev, A)
x[t] ~ DiscreteTransition(s[t], B)

# From POMDP
current_state ~ DiscreteTransition(previous_state, B, previous_control)
current_y ~ DiscreteTransition(current_state, A)
```

## Vague/Uninformative Priors

```julia
# From initialization
q(A) = vague(DirichletCollection, (3, 3))
q(s) = vague(Categorical, 3)
q(σ) = vague(InverseGamma)
μ(α) = vague(NormalMeanVariance)
```

## Alternative Parameterizations

```julia
# Normal with named parameters
Normal(mean = 0.0, variance = 1.0)
Normal(mean = 0.0, precision = 1.0)

# Direct construction
NormalMeanVariance(0.0, 1.0)
NormalMeanPrecision(0.0, 1.0)
GammaShapeRate(a, b)
GammaShapeScale(a, θ)
```

## Statistics Functions

```julia
# From Linear Regression results
mean(result.posteriors[:a])
var(result.posteriors[:a])
std(result.posteriors[:a])
mean_var(result.posteriors[:a])  # Returns tuple
mean_std(result.posteriors[:a])  # Returns tuple
mean_precision(q(x))  # For autoupdates

# For Gamma
shape(q(τ_x))
rate(q(τ_x))

# Sampling
rand(result.posteriors[:a], 100)  # 100 samples
```

## Conjugate Pairs

| Likelihood | Prior | Closed-Form Update |
|------------|-------|-------------------|
| `Bernoulli(θ)` | `Beta(a, b)` | ✓ |
| `Normal(μ, σ)` | `Normal(μ₀, σ₀)` | ✓ |
| `Normal(μ, τ)` | `Gamma(α, β)` | ✓ |
| `Categorical(π)` | `Dirichlet(α)` | ✓ |
| `MvNormal(μ, Σ)` | `MvNormal`, `InverseWishart` | ✓ |
| `DiscreteTransition` | `DirichletCollection` | ✓ |

## Related Examples

- `scripts/Basic Examples/Coin Toss Model/` - Beta-Bernoulli
- `scripts/Basic Examples/Hidden Markov Model/` - DirichletCollection
- `scripts/Basic Examples/Kalman filtering and smoothing/` - MvNormal
- `scripts/Basic Examples/Bayesian Linear Regression/` - InverseGamma, InverseWishart
