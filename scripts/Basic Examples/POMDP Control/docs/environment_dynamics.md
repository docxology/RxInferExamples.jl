# Environment Dynamics

## WindyGridWorld

The environment is a **5×5 grid** with wind that affects transitions.

### Configuration

```julia
grid_params = (
    grid_size = 5,
    wind = (0, 1, 1, 1, 0),  # Wind strength per column
    goal = (4, 3),
    start = (1, 1)
)
```

### Grid Layout

```
Y
5 │ · · · · ·    Wind: ↑ ↑ ↑
4 │ · · · · ·         0 1 1 1 0
3 │ · · · ★ ·    ★ = Goal (4,3)
2 │ · · · · ·    
1 │ ○ · · · ·    ○ = Start (1,1)
  └─────────────
    1 2 3 4 5  X
```

### Wind Effects

Columns 2, 3, 4 have **upward wind** (strength = 1):
- Agent moves intended direction PLUS wind pushback
- Example: At (2,1), action "right" → (3, 2) (wind pushes up)

## State Representation

### State Space

- **25 states**: Flattened grid positions
- **Index mapping**: `index = (x-1) * grid_size + y`

```julia
function grid_location_to_index(pos::Tuple{Int,Int})
    x, y = pos
    return (x - 1) * grid_params.grid_size + y
end

function index_to_grid_location(index::Int)
    x = div(index - 1, grid_params.grid_size) + 1
    y = mod(index - 1, grid_params.grid_size) + 1
    return (x, y)
end
```

### One-Hot Encoding

States/observations are one-hot vectors of length 25:

```julia
function index_to_one_hot(index::Int)
    v = zeros(grid_params.grid_size^2)
    v[index] = 1.0
    return v
end
```

## Action Space

### Actions

| Index | Name | Movement (Δx, Δy) |
|-------|------|-------------------|
| 1 | Up | (0, +1) |
| 2 | Right | (+1, 0) |
| 3 | Down | (0, -1) |
| 4 | Left | (-1, 0) |

### Movement Execution

```julia
ACTION_MOVEMENTS = Dict(
    ACTION_UP    => (0, 1),
    ACTION_RIGHT => (1, 0),
    ACTION_DOWN  => (0, -1),
    ACTION_LEFT  => (-1, 0)
)
```

## Transition Dynamics

### Without Wind (Columns 1, 5)

```
new_x = clamp(x + Δx, 1, 5)
new_y = clamp(y + Δy, 1, 5)
```

### With Wind (Columns 2, 3, 4)

```
new_x = clamp(x + Δx, 1, 5)
new_y = clamp(y + Δy + wind[new_x], 1, 5)
```

### Boundary Conditions

Agent cannot leave the grid; attempts to move outside result in staying at edge.

## True Transition Matrix

For action "right" (index 2), the true B matrix would be:

```
B[new_state, old_state, RIGHT] = 1.0 where:
  (new_x, new_y) = transition(old_x, old_y, RIGHT)
```

**Wind creates non-intuitive transitions:**
- From (1,1) with action RIGHT: ends at (2,2) not (2,1)
- The agent must LEARN this through experience

## RxEnvironments Integration

### Environment Creation

```julia
mutable struct WindyGridWorld{N}
    position::Tuple{Int,Int}
end

function create_environment()
    env = WindyGridWorld{grid_params.grid_size}(grid_params.start)
    rx_actor = create(env)
    rx_agent = add!(rx_actor, ActionEntity())
    return env, rx_agent
end
```

### Receiving and Actions

```julia
# Send action to environment
send!(env, rx_agent, movement)

# Environment updates position and emits observation
RxEnvironments.update!(env, agent, action)
RxEnvironments.what_to_send_to_actor(env, entity) = position
```

### Observation Stream

```julia
observations = keep(Any)
RxEnvironments.subscribe_to_observations!(rx_agent, observations)

# Access latest observation
last_obs = RxEnvironments.data(last(observations))
```

## Optimal Policy

Given full knowledge of wind, optimal policy from (1,1) to (4,3):

1. **Right** → (2, 2) [wind +1]
2. **Right** → (3, 3) [wind +1]
3. **Right** → (4, 4) [wind +1]
4. **Down** → (4, 3) ✓ [no wind in column 4... wait, there is]

Actually column 4 HAS wind, so:
- From (4,4), Down → (4-0, 4-1+1) = (4, 4) [stays!]

This makes the problem harder - agent must find alternate routes or compensate.
