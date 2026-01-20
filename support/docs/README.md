# RxInfer.jl Documentation

Comprehensive reference for working with RxInfer.jl, with real syntax from working examples in this repository.

## Quick Reference

| Topic | File | Key Concepts |
|-------|------|--------------|
| **Core** | [core_concepts.md](core_concepts.md) | `infer()`, free energy, conjugate pairs |
| **Models** | [model_specification.md](model_specification.md) | `@model`, `@constraints`, `@initialization` |
| **Inference** | [inference.md](inference.md) | VMP, streaming, `@autoupdates` |
| **Distributions** | [distributions.md](distributions.md) | Normal, Gamma, Dirichlet, etc. |
| **Nodes** | [factor_nodes.md](factor_nodes.md) | DiscreteTransition, DirichletCollection |
| **SSM** | [state_space_models.md](state_space_models.md) | Kalman, HMM, smoothing |
| **Active Inference** | [active_inference.md](active_inference.md) | POMDP, goal-directed control |
| **Performance** | [performance.md](performance.md) | Optimization, stack depth |

## Minimal Working Example

```julia
using RxInfer

@model function coin_model(y, a, b)
    θ ~ Beta(a, b)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

result = infer(
    model = coin_model(a = 4.0, b = 8.0), 
    data  = (y = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],)
)

θestimated = result.posteriors[:θ]
```

## Essential Macros

| Macro | Purpose | Example |
|-------|---------|---------|
| `@model` | Define model | `@model function my_model() ... end` |
| `@constraints` | Factorization | `q(a, b) = q(a)q(b)` |
| `@initialization` | Initial values | `q(τ) = vague(Gamma)` |
| `@meta` | Approximations | `smooth_min() -> Linearization()` |
| `@autoupdates` | Streaming | `m_x = mean_precision(q(x))` |

## Source Examples

All documentation references working scripts in this repository:

- `scripts/Basic Examples/Coin Toss Model/` - Beta-Bernoulli
- `scripts/Basic Examples/Hidden Markov Model/` - Discrete SSM
- `scripts/Basic Examples/Kalman filtering and smoothing/` - Continuous SSM
- `scripts/Basic Examples/Bayesian Linear Regression/` - Hierarchical
- `scripts/Basic Examples/POMDP Control/` - Active inference

## Key Links

- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl)
- [ReactiveMP.jl](https://github.com/ReactiveBayes/ReactiveMP.jl)
