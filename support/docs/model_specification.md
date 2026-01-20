# Model Specification

Guide to specifying probabilistic models in RxInfer using the `@model` macro, with real syntax from working examples.

## The @model Macro

Models are defined using the `@model` macro:

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
| `x := f(...)` | Deterministic | `s[i] := f(x[i], w[i])` |
| `y .~ Distribution()` | Vectorized | `y .~ Normal(mean = a .* x .+ b, variance = 1.0)` |

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

## State Space Models

### Linear Gaussian SSM (Kalman Filter)

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

# Usage with matrix parameters
θ = π / 35
A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
B = diageye(2)
Q = 25.0 * diageye(2)
P = diageye(2)
x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))

result = infer(
    model = rotate_ssm(x0=x0, A=A, B=B, P=P, Q=Q), 
    data = (y = y,),
    free_energy = true
)
```

### Hidden Markov Model

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

## Constraints (@constraints)

Define factorization for variational inference:

```julia
# From HMM example
@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end

# From Kalman identification problem
constraints = @constraints begin 
    q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
end

# Mean-field shorthand
constraints = MeanField()
```

## Initialization (@initialization)

Set initial marginals/messages for VMP:

```julia
# From HMM example
imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(B) = vague(DirichletCollection, (3, 3)) 
    q(s) = vague(Categorical, 3)
end

# From Kalman example - message initialization
init = @initialization begin
    μ(x) = xinit  # Vector of initial messages
    μ(w) = winit
    q(τ_x) = GammaShapeRate(a_x, b_x)
    q(τ_w) = GammaShapeRate(a_w, b_w)
    q(τ_y) = GammaShapeRate(a_y, b_y)
end

# Inline initialization
results = infer(
    model = linear_regression(), 
    data  = (y = y_data, x = x_data), 
    initialization = @initialization(μ(b) = NormalMeanVariance(0.0, 100.0))
)
```

## Return Variables

Control which posteriors are returned:

```julia
# Keep last iteration only
ireturnvars = (
    A = KeepLast(),
    B = KeepLast(),
    s = KeepLast()
)

# Alternative syntax
returnvars = (a = KeepLast(), b = KeepLast())

# Keep all iterations
returnvars = KeepLast()  # For all variables
```

## Hierarchical Models

```julia
# From Bayesian Linear Regression (partially pooled)
@model function partially_pooled(patient_codes, weeks, data)
    μ_α ~ Normal(mean = 0.0, var = 250000.0)
    μ_β ~ Normal(mean = 0.0, var = 9.0)
    σ_α ~ Gamma(shape = 1.75, scale = 45.54)
    σ_β ~ Gamma(shape = 1.75, scale = 1.36)

    n_patients = length(unique(patient_codes))

    local α
    local β

    for i in 1:n_patients
        α[i] ~ Normal(mean = μ_α, precision = σ_α)
        β[i] ~ Normal(mean = μ_β, precision = σ_β)
    end

    σ ~ Gamma(shape = 1.75, scale = 45.54)
    
    local FVC_est

    for i in 1:length(patient_codes)
        FVC_est[i] ~ α[patient_codes[i]] + β[patient_codes[i]] * weeks[i]
        data[i] ~ Normal(mean = FVC_est[i], precision = σ)
    end
end
```

## Deterministic Transformations

```julia
# From Kalman identification problem
@model function identification_problem(f, y, ...)
    x[i] ~ Normal(mean = x_i_min, precision = τ_x)
    w[i] ~ Normal(mean = w_i_min, precision = τ_w)
    s[i] := f(x[i], w[i])  # Deterministic node
    y[i] ~ Normal(mean = s[i], precision = τ_y)
end
```

## Meta Information (@meta)

For nonlinear functions requiring approximation:

```julia
# From Kalman example with smooth_min
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

- `scripts/Basic Examples/Coin Toss Model/`
- `scripts/Basic Examples/Hidden Markov Model/`
- `scripts/Basic Examples/Kalman filtering and smoothing/`
- `scripts/Basic Examples/Bayesian Linear Regression/`
