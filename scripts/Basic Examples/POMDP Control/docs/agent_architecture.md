# Agent Architecture

## PODMPAgent Structure

The agent orchestrates perception, inference, and action in [`src/Agent.jl`](../src/Agent.jl):

```julia
mutable struct PODMPAgent
    # Environment references
    environment::Any
    rx_agent::Any
    observations::Any
    
    # Learned model parameters
    p_A::Any  # Observation model posterior
    p_B::Any  # Transition model posterior
    
    # State beliefs
    p_s::Any  # Current state belief
    
    # Goal specification
    goal::Any
    
    # Policy and action tracking
    current_policy::Any
    prev_action::Vector{Float64}
    
    # Statistics
    step_count::Int
end
```

## Control Loop

The agent operates in a **perceive-act-infer** cycle:

```
┌─────────────────────────────────────────────────────┐
│                  CONTROL LOOP                       │
│                                                     │
│  ┌─────────┐     ┌─────────┐     ┌─────────────┐   │
│  │  ACT    │────►│ OBSERVE │────►│   INFER     │   │
│  │(policy) │     │ (o_t)   │     │(update all) │   │
│  └────▲────┘     └─────────┘     └──────┬──────┘   │
│       │                                  │          │
│       └──────────────────────────────────┘          │
│                new policy π                         │
└─────────────────────────────────────────────────────┘
```

### Step Order

The `step!` function follows this sequence:

```julia
function step!(agent, remaining_horizon)
    # 1. ACT: Execute action from current policy
    action_idx = mode(first(agent.current_policy))
    send!(agent.environment, agent.rx_agent, movement)
    
    # 2. OBSERVE: Get observation after action
    last_obs = RxEnvironments.data(last(agent.observations))
    
    # 3. CHECK: Goal reached?
    if last_obs == grid_params.goal
        return true
    end
    
    # 4. INFER: Update beliefs and plan
    inference_result = infer(model = pomdp_model(...), ...)
    
    # 5. LEARN: Update parameters
    agent.p_s = last(inference_result.posteriors[:current_state])
    agent.current_policy = last(inference_result.posteriors[:controls])
    agent.p_A = last(inference_result.posteriors[:A])
    agent.p_B = last(inference_result.posteriors[:B])
    
    return false
end
```

## Initialization

### Agent Creation

```julia
function PODMPAgent(env, rx_agent)
    # 1. Initialize model priors
    p_A, p_B = create_initial_priors()
    
    # 2. Initial state belief at start position
    start_idx = grid_location_to_index(grid_params.start)
    p_s = Categorical(index_to_one_hot(start_idx))
    
    # 3. Goal prior
    goal = create_goal_prior(grid_params.goal)
    
    # 4. Subscribe to observations
    observations = keep(Any)
    RxEnvironments.subscribe_to_observations!(rx_agent, observations)
    
    # 5. Initial "down" policy
    initial_policy = [Categorical([0.0, 0.0, 1.0, 0.0])]
    
    return PODMPAgent(...)
end
```

### Why "Down" Initial Policy?

From start position (1,1), "down" is neutral:
- Cannot go further down (boundary)
- Doesn't commit to any particular direction
- Gives agent first observation without moving

## Reset Protocol

Between experiments, partial state is reset:

```julia
function reset!(agent)
    reset_env!(agent.environment)
    agent.p_s = Categorical(index_to_one_hot(start_idx))
    agent.current_policy = [Categorical([0.0, 0.0, 1.0, 0.0])]
    agent.step_count = 0
    # NOTE: p_A and p_B are NOT reset - learning persists!
end
```

**Key Design Decision**: Learning parameters persist across experiments, enabling cumulative learning.

## Experiment Execution

```julia
function run_experiment!(agent, T)
    reset!(agent)
    
    for t in 1:T
        goal_reached = step!(agent, T - t)
        if goal_reached
            return true
        end
    end
    
    return false  # Didn't reach goal in T steps
end
```

## Logging

The agent uses structured logging via [`src/Utils.jl`](../src/Utils.jl):

```julia
function rxlog(level::String, message::String)
    timestamp = Dates.format(Dates.now(), "HH:MM:SS.sss")
    log_line = "[$timestamp] [$level] $message"
    println(log_line)
    
    # Also write to log file
    open(LOG_FILE, "a") do io
        println(io, log_line)
    end
end
```

### Log Levels

| Level | Usage |
|-------|-------|
| `info` | High-level events (experiment start/end) |
| `debug` | Step-by-step details (actions, observations) |
| `warn` | Unexpected but recoverable events |
| `error` | Failures requiring attention |

## Performance Considerations

### Memory

- A matrix: 25×25 = 625 floats
- B matrix: 25×25×4 = 2,500 floats
- State belief: 25 floats
- Total per agent: ~25 KB

### Computation per Step

- Model construction: O(1)
- Inference (20 iterations): ~200K ops
- Action selection: O(4)
- Total: ~1-5ms per step on modern CPU

### Scalability

| Grid Size | States | A Size | B Size | Time/Step |
|-----------|--------|--------|--------|-----------|
| 5×5 | 25 | 625 | 2,500 | ~2ms |
| 10×10 | 100 | 10K | 40K | ~50ms |
| 20×20 | 400 | 160K | 640K | ~1s |

Current implementation optimized for demonstration, not large-scale MDPs.
