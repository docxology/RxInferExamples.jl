# Inference

Running Bayesian inference with RxInfer.jl, with real syntax from working examples.

## Basic Inference

The `infer()` function is the entry point for running inference.

```julia
# From Coin Toss example
result = infer(
    model = coin_model(a = 4.0, b = 8.0), 
    data  = (y = dataset,)
)

θestimated = result.posteriors[:θ]
```

## Advanced Inference Configuration

### Full Parameter Set

```julia
# From HMM example
result = infer(
    model          = hidden_markov_model(), 
    data           = (x = x_data,),
    constraints    = hidden_markov_model_constraints(),
    initialization = imarginals, 
    returnvars     = ireturnvars, 
    iterations     = 20, 
    free_energy    = true,
    options        = (limit_stack_depth = 500,)
)
```

| Parameter | Description |
|-----------|-------------|
| `model` | The instantiated model object |
| `data` | NamedTuple of data (observations) |
| `constraints` | Factorization constraints for VMP |
| `initialization` | Initial values for marginals or messages |
| `returnvars` | Specification of what to return (saves memory) |
| `iterations` | Number of VMP iterations per data batch |
| `free_energy` | Boolean, whether to compute Bethe Free Energy |
| `autoupdates` | For reactive/streaming inference (see below) |

## Reactive / Streaming Inference

RxInfer excels at online, streaming inference using `@autoupdates` and RxEnvironments.

### The @autoupdates Macro

This macro defines how priors for the *next* time step are updated based on the posteriors of the *current* time step.

```julia
# From Kalman Filter (streaming)
autoupdates = @autoupdates begin 
    # Current posterior q(x) becomes next prior x_prior
    # mean_precision extracts parameters from the distribution
    m_x_0, τ_x_0 = mean_precision(q(x))
    
    # Update noise parameters
    a_x = shape(q(τ_x)); b_x = rate(q(τ_x))
end
```

### Running the Engine

For streaming, you create a reactive engine that listens for data.

```julia
# Initialize Engine
engine = infer(
    model         = methods_of_model(...),
    autoupdates   = autoupdates,
    autostart     = false  # Don't start until data subscription
)

# Subscribe to Results
RxInfer.subscribe!(engine.posteriors[:x]) do marginal
    println("New state estimate: ", mean(marginal))
end

# Feed Data
RxInfer.next!(engine, (y = new_observation,))
```

## Accessing Results

### Posteriors

The `result.posteriors` field is a NamedTuple.

```julia
# Univariate
θ = result.posteriors[:θ]  # Vector{Beta} (one per iteration/time)

# Analysis
m = mean.(θ)
v = var.(θ)

# Multivariate
s = result.posteriors[:s]  # Vector{Categorical}
probs = ReactiveMP.probvec.(s)  # Get probability vectors
```

### Free Energy

If `free_energy=true`, `result.free_energy` contains the history of the Bethe Free Energy (BFE). Decreasing BFE indicates convergence.

```julia
using Plots
plot(result.free_energy, label="BFE")
```

### Troubleshooting Convergence

- **BFE Oscillates**: Learning rate might be too high (if using sophisticated update rules) or the model is unidentifiable.
- **BFE Increases**: Constraints might be violating the model structure, or numerical instability.
- **NaN Results**: Usually initialization issues. Check your `@initialization` block. Ensure priors are not too vague (infinite variance).
