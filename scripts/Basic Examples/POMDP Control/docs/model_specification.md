# Model Specification

## RxInfer Model Definition

The POMDP model is defined using RxInfer's `@model` macro in [`src/Model.jl`](../src/Model.jl):

```julia
@model function pomdp_model(
    p_A, p_B, p_goal, p_control,
    previous_control, p_previous_state,
    current_y, future_y, T, m_A, m_B
)
    # Parameter inference
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)
    
    # Inference-as-planning loop
    prev_state = current_state
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    
    # Goal prior
    s[end] ~ p_goal
end
```

## Model Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `p_A` | `DirichletCollection` | Prior on observation matrix (learned) |
| `p_B` | `DirichletCollection` | Prior on transition matrix (learned) |
| `p_goal` | `Categorical` | Goal state prior (fixed, one-hot at goal) |
| `p_control` | `Categorical` | Prior on actions (vague = uniform) |
| `previous_control` | `Vector{Float64}` | One-hot previous action |
| `p_previous_state` | `Categorical` | Prior state belief |
| `current_y` | `Vector{Float64}` | Current observation (one-hot position) |
| `future_y` | `Vector{Missing}` | Future observations (missing = to be inferred) |
| `T` | `Int` | Planning horizon |
| `m_A` | `Matrix{Float64}` | Mean of A for planning |
| `m_B` | `Array{Float64,3}` | Mean of B for planning |

## Prior Distributions

### A Matrix Prior (Observation Model)

```julia
p_A = DirichletCollection(diageye(n_states) .+ 0.1)
```

- **Structure**: Weakly peaked on diagonal
- **Interpretation**: Prior belief that observations reveal true state  
- **Learning**: Sharpens as agent receives more observations

### B Matrix Prior (Transition Model)

```julia
p_B = DirichletCollection(ones(n_states, n_states, n_actions))
```

- **Structure**: Uniform across all transitions
- **Interpretation**: No prior knowledge of dynamics
- **Learning**: Discovers wind effects through experience

### Goal Prior

```julia
goal_idx = grid_location_to_index(grid_params.goal)
p_goal = Categorical(index_to_one_hot(goal_idx))
```

- **Structure**: Delta distribution at goal position (4,3)
- **Role**: Provides pragmatic drive toward goal in EFE

### Control Prior

```julia
p_control = vague(Categorical, 4)  # Uniform over 4 actions
```

- **Structure**: Maximum entropy (uniform)
- **Role**: No inherent action preference

## Factorization Constraints

The `@constraints` block specifies factorization for message passing:

```julia
@constraints function pomdp_constraints()
    q(previous_state, previous_control, current_state, B) = 
        q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls)q(B)
    q(s, future_y, A) = q(s, future_y)q(A)
end
```

### Interpretation

These constraints ensure **mean-field factorization** between:
- State/control variables and model parameters (A, B)
- This allows parameters to be updated independently from states

## Initialization

```julia
@initialization begin
    q(A) = DirichletCollection(diageye(n_states) .+ 0.1)
    q(B) = DirichletCollection(ones(n_states, n_states, n_actions))
end
```

This sets initial variational posteriors for iterative message passing.

## Data Specification

When calling `infer()`:

```julia
data = (
    previous_control = agent.prev_action,        # Known: previous action
    current_y = obs_one_hot,                     # Known: current observation
    future_y = UnfactorizedData(fill(missing, T))  # Unknown: future observations
)
```

- **Known data**: Clamped in factor graph
- **Missing data**: Inferred via message passing (planning)

## Inference Configuration

```julia
inference_result = infer(
    model = pomdp_model(...),
    data = (...),
    constraints = pomdp_constraints(),
    initialization = make_pomdp_initialization(),
    iterations = 20  # VMP iterations
)
```

The 20 iterations allow beliefs to converge through variational message passing.
