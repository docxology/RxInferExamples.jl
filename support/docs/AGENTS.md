# RxInfer Documentation - Agent Guide

Reference docs for RxInfer.jl with real syntax from working examples.

## Files

| File | Content |
|------|---------|
| [core_concepts.md](core_concepts.md) | `infer()`, free energy, conjugate pairs |
| [model_specification.md](model_specification.md) | `@model`, `@constraints`, `@initialization`, `@meta` |
| [inference.md](inference.md) | VMP, streaming, `@autoupdates` |
| [distributions.md](distributions.md) | All distribution types |
| [factor_nodes.md](factor_nodes.md) | DiscreteTransition, DirichletCollection |
| [state_space_models.md](state_space_models.md) | Kalman, HMM, smoothing |
| [active_inference.md](active_inference.md) | POMDP control patterns |
| [performance.md](performance.md) | Optimization tips |

## Key Patterns

```julia
# Basic inference
result = infer(model = my_model(), data = (y = data,))

# VMP with constraints
result = infer(
    model = model(),
    data = data,
    constraints = MeanField(),
    iterations = 20
)

# Access results
mean(result.posteriors[:θ])
```
