# Performance Tips for RxInfer.jl

Optimizing probabilistic models for speed and memory efficiency.

## 1. Vectorization

**Vectorized operations are significantly faster** than looping over scalar nodes in Julia, and this extends to graph construction in RxInfer.

### Bad (Scalar Loop)

```julia
# 1000 individual scalar nodes
for i in 1:1000
    y[i] ~ Normal(x[i], 1.0)
end
```

**Impact**: Creates 1000 separate nodes in the factor graph. High overhead for graph construction and message scheduling.

### Good (Vectorized)

```julia
# 1 node processing a vector of 1000 elements
y ~ MvNormal(x, diageye(1000))
```

**Impact**: Creates a single node. Message passing is batched using optimized linear algebra operations (BLAS/LAPACK).

## 2. Conjugate Priors

Whenever possible, use **conjugate priors**.

- **Gaussian-Gaussian**: Analytical, fast, exact.
- **Beta-Bernoulli**: Analytical, fast.
- **Dirichlet-Categorical**: Analytical, fast.

**Non-conjugate pairs** require approximation methods (Variational Message Passing, Belief Propagation with approximations) which are iterative and computationally more expensive.

## 3. Message Scheduling

For cyclic graphs (common in variational inference), the order of message updates matters.

- **Default**: RxInfer attempts a reasonable default schedule.
- **Custom Schedule**: In complex models, defining a manual schedule can improve convergence speed.

## 4. Avoiding `Any`-typed Containers

Julia performance relies on type stability.

- Ensure your data containers are strictly typed (e.g., `Vector{Float64}` instead of `Vector{Any}`).
- Use `fill(missing, N)` for missing data arrays to ensure the container type is efficient.

## 5. Pre-allocation

If running reactive inference (streaming) in a tight loop:

- Reuse memory where possible.
- Avoid recreating the model structure every millisecond if the structure is static; use the `RxInferenceEngine` API to update data only.

## 6. Matrix Operations

- **Diagonal Matrices**: Use `diageye(n)` or `Diagonal` from `LinearAlgebra`. This allows Julia to use specialized sparse algorithms.

  ```julia
  # Fast
  Σ ~ InverseWishart(v, Diagonal(ones(d)))
  
  # Slower (Dense)
  Σ ~ InverseWishart(v, Matrix(I, d, d))
  ```

## 7. Initialization

Good initialization speeds up convergence in variational inference.

- If you have a heuristic guess for the posterior, initialize the variational distribution `q` close to that guess.
- Bad initialization can lead to getting stuck in local optima, requiring many more iterations to escape (or failing to converge).
