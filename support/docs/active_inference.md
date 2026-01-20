# Active Inference

Active inference and POMDP control in RxInfer.jl, with real syntax from the POMDP Control example.

## POMDP Model Structure

```julia
# From POMDP Control example
@model function pomdp_model(p_A, p_B, p_goal, p_control, previous_control, p_previous_state, current_y, future_y, T, m_A, m_B)
    # Instantiate all model parameters with priors
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    
    # Parameter inference
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    # Inference-as-planning
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    # Goal prior initialization
    s[end] ~ p_goal
end
```

## Constraints for Active Inference

```julia
# From POMDP Control
constraints = @constraints begin
    q(previous_state, previous_control, current_state, B) = q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls), q(B)
    q(s, future_y, A) = q(s, future_y), q(A)
end
```

## Initialization

```julia
# From POMDP Control
init = @initialization begin
    q(A) = DirichletCollection(diageye(25) .+ 0.1)
    q(B) = DirichletCollection(ones(25, 25, 4))
end

# Priors
p_A = DirichletCollection(diageye(25) .+ 0.1)  # Observation model
p_B = DirichletCollection(ones(25, 25, 4))      # Transition model
```

## Goal Specification

```julia
# One-hot encoding for goal state
function index_to_one_hot(index::Int)
    return [i == index ? 1.0 : 0.0 for i in 1:25]
end

goal = Categorical(index_to_one_hot(grid_location_to_index((4, 3))))
```

## Inference Loop

```julia
# From POMDP Control - complete control loop
for t in 1:T
    # Get observation
    last_observation = index_to_one_hot(grid_location_to_index(position))
    
    # Perform inference
    inference_result = infer(
        model = pomdp_model(
            p_A = p_A,
            p_B = p_B,
            T = max(T - t, 1),
            p_previous_state = p_s,
            p_goal = goal,
            p_control = vague(Categorical, 4),
            m_A = mean(p_A),
            m_B = mean(p_B)
        ),
        data = (
            previous_control = prev_u,
            current_y = last_observation,
            future_y = UnfactorizedData(fill(missing, max(T - t, 1)))
        ),
        constraints = constraints,
        initialization = init,
        iterations = 10
    )
    
    # Update beliefs
    p_s = last(inference_result.posteriors[:current_state])
    policy = last(inference_result.posteriors[:controls])

    # Update model parameters
    p_A = last(inference_result.posteriors[:A])
    p_B = last(inference_result.posteriors[:B])
    
    # Execute action
    current_action = mode(first(policy))
    execute_action(current_action)
end
```

## Action Selection

```julia
# From POMDP Control
current_action = mode(first(policy))
if current_action == 1
    send!(env, agent, (0, 1))   # Move up 
    prev_u = [1.0, 0.0, 0.0, 0.0]
elseif current_action == 2
    send!(env, agent, (1, 0))   # Move right
    prev_u = [0.0, 1.0, 0.0, 0.0]
elseif current_action == 3
    send!(env, agent, (0, -1))  # Move down
    prev_u = [0.0, 0.0, 1.0, 0.0]
elseif current_action == 4
    send!(env, agent, (-1, 0))  # Move left
    prev_u = [0.0, 0.0, 0.0, 1.0]
end
```

## Generative Model Matrices

### A Matrix (Observation Likelihood)

```julia
# P(o|s) - maps states to observations
# Diagonal = perfect observation
p_A = DirichletCollection(diageye(25) .+ 0.1)
```

### B Matrix (Transition Model)

```julia
# P(s'|s,a) - 3D tensor for state transitions given actions
# Shape: (n_states, n_states, n_actions)
p_B = DirichletCollection(ones(25, 25, 4))
```

### DiscreteTransition

```julia
# State transition with action
current_state ~ DiscreteTransition(previous_state, B, previous_control)

# Observation generation
current_y ~ DiscreteTransition(current_state, A)
```

## Handling Missing Future Data

```julia
# Future observations are unknown during planning
future_y = UnfactorizedData(fill(missing, horizon))
```

## RxEnvironments Integration

```julia
using RxEnvironments

# Create environment
env = RxEnvironment(WindyGridWorld((0, 1, 1, 1, 0), [], (4, 3)))
agent = add!(env, WindyGridWorldAgent((1, 1)))

# Subscribe to observations
observations = keep(Any)
RxEnvironments.subscribe_to_observations!(agent, observations)

# Send actions
send!(env, agent, (0, 1))  # Move

# Get observation
last_observation = RxEnvironments.data(last(observations))
```

## Related Examples

- `scripts/Basic Examples/POMDP Control/`
- `scripts/drone/reactive-drone.jl`
- `scripts/pendulum/reactive-pendulum.jl`
- `scripts/Advanced Examples/Active Inference Mountain car/`
