# Performance Optimization

Tips for maximizing RxInfer.jl performance, with real patterns from working examples.

## Stack Depth for Large Models

```julia
# From Linear Regression hierarchical example
result = infer(
    model = partially_pooled(patient_codes = patient_codes, weeks = weeks),
    data = (data = FVC_obs,),
    options = (limit_stack_depth = 500,),  # Essential for >100 observations
    constraints = constraints,
    initialization = init,
    iterations = 100
)
```

## Iteration Count

```julia
# Simple models: 1-5 iterations
result = infer(model = coin_model(), data = data)  # Default: 1

# Complex VMP: 10-50 iterations
result = infer(
    model = hidden_markov_model(),
    iterations = 20,
    free_energy = true  # Monitor convergence
)

# Hierarchical models: 50-100 iterations
result = infer(
    model = partially_pooled(...),
    iterations = 100
)
```

## Free Energy Monitoring

```julia
# From Linear Regression
plot(results.free_energy[2:end],  # Skip init step
     title="Free energy", 
     xlabel="Iteration",
     ylabel="Free energy [nats]")

# Check convergence
if results.free_energy[end] - results.free_energy[end-1] > 0.1
    println("Warning: May not have converged")
end
```

## Initialization Quality

```julia
# From Kalman - good initialization speeds convergence
xinit = map(r -> NormalMeanPrecision(r, τ_x_0), reverse(range(-60, -20, length = n)))
winit = map(r -> NormalMeanPrecision(r, τ_w_0), range(20, 60, length = n))

init = @initialization begin
    μ(x) = xinit  # Informed initial messages
    μ(w) = winit
    q(τ_x) = GammaShapeRate(a_x, b_x)
end
```

## Constraint Selection

```julia
# Full mean-field (fastest, least accurate)
constraints = MeanField()

# Structured VMP (slower, more accurate)
@constraints function structured_constraints()
    q(x0, w0, x, w, τ_x, τ_w, τ_y, s) = q(x, x0, w, w0, s)q(τ_w)q(τ_x)q(τ_y)
end
```

## Streaming for Large Data

```julia
# From Kalman - @autoupdates for online inference
autoupdates = @autoupdates begin 
    m_x_0, τ_x_0 = mean_precision(q(x))
    m_w_0, τ_w_0 = mean_precision(q(w))
end

engine = infer(
    model = streaming_model(...),
    autoupdates = autoupdates,
    keephistory = 1000,
    autostart = true
)
```

## Vague Priors

```julia
# Use vague() for uninformative initialization
q(A) = vague(DirichletCollection, (3, 3))
q(s) = vague(Categorical, 3)
q(σ) = vague(InverseGamma)
μ(α) = vague(NormalMeanVariance)
```

## ReturnVars Optimization  

```julia
# Only return what you need
returnvars = (a = KeepLast(), b = KeepLast())

# For debugging, keep all
returnvars = KeepLast()  # All variables, last iteration
```

## Benchmarking

```julia
using BenchmarkTools

@benchmark infer(
    model = $rotate_ssm(x0=$x0, A=$A, B=$B, P=$P, Q=$Q), 
    data  = (y = $y,)
)
```

## Common Performance Issues

| Issue | Solution |
|-------|----------|
| Stack overflow | Add `options = (limit_stack_depth = 500,)` |
| Slow convergence | Better initialization, more iterations |
| Memory issues | Use `returnvars = KeepLast()` |
| Oscillating F | Check constraints, try structured VMP |

## Related Examples

- `scripts/Basic Examples/Kalman filtering and smoothing/` - Streaming
- `scripts/Basic Examples/Bayesian Linear Regression/` - Large hierarchical
- `scripts/Basic Examples/Hidden Markov Model/` - VMP iterations
