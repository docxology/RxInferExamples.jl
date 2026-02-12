# Debugging RxInfer Models

This guide covers common issues encountered when building and running active inference models in RxInfer.jl and provides strategies for resolution.

## Common Errors

### `PosDefException`

**Error**: `PosDefException: matrix is not positive definite`
**Cause**: Covariance matrices (in Gaussian distributions) must be positive definite. This often happens when:

1. Precision becomes too high (variance near zero).
2. Numerical instability in recursive updates.
3. Improper initialization of priors.

**Fix**:

- **Regularization**: Add a small "jitter" to the diagonal of your covariance matrices.

  ```julia
  # Add small noise to diagonal
  jitter = 1e-6 * diageye(d)
  Σ_safe = Σ + jitter
  ```

- **Priors**: Ensure priors are not too tight (low variance) if the data is noisy.
- **Micro-motion**: If using a `Matrix` node, ensure the input matrix is strictly positive definite.

### `UndefVarError` in `@model`

**Error**: `UndefVarError: x not defined` inside `@model` macro.
**Cause**: The `@model` macro creates a closed scope. Variables defined outside the macro are not automatically visible unless passed as arguments or defined as constants/globals (not recommended).
**Fix**:

- Pass external parameters as arguments to the model function.

  ```julia
  # BAD
  dim = 5
  @model function my_model(...)
      x ~ MvNormal(zeros(dim), I) # dim is not visible
  end
  
  # GOOD
  @model function my_model(dim, ...)
      x ~ MvNormal(zeros(dim), I)
  end
  ```

### `MethodError: no method matching ...`

**Cause**: Often due to mismatch between the node type and the message passing rules available.
**Fix**:

- Check if the specific combination of parent/child distributions has a valid conjugate update rule.
- Use `@meta` to specify an approximation method (e.g., Linearization, Unscented) if no analytical rule exists.

## GraphPPL Constraints

### Recursive Dependencies

RxInfer requires a Directed Acyclic Graph (DAG) for the model structure *within a single time step*.

- **Cyclic dependencies** (e.g., `a` depends on `b`, `b` depends on `a`) are not allowed in the static structure.
- **Time-step recursion** (e.g., `x[t]` depends on `x[t-1]`) is allowed and handled by variable indexing.

### Deterministic Nodes

Deterministic nodes (defined with `=`) propagate messages differently than stochastic nodes (defined with `~`).

- Use `~ Deterministic(f(x))` if you need to treat a deterministic transformation as a probabilistic node (Dirac delta) to facilitate message passing with approximation methods.

## Debugging Workflow

1. **Simplify**: Reduce the model to the smallest component that reproduces the error.
2. **Visualize**: Use `RxVisualization.plot_graph(model)` (if available) or draw the factor graph to check connections.
3. **Inspect Messages**:
    - Use data logging to inspect the posterior beliefs at each step.
    - Check for `NaN` or `Inf` values in the results.

## Advanced: `@meta` Debugging

When using `@meta` for approximations:

```julia
@meta function MyMeta()
    model(x, y) -> Linearization()
end
```

- Ensure the function names in `@meta` match the nodes in `@model`.
- If an approximation fails, try a more robust one (e.g., switching from `Linearization` to `Unscented`).
