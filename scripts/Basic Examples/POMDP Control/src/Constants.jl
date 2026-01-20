# Action Constants for POMDP Control
# Grid and experiment parameters are loaded from config.toml via Config.jl
# See: docs/configuration.md

# Action indices
const ACTION_UP = 1
const ACTION_RIGHT = 2
const ACTION_DOWN = 3
const ACTION_LEFT = 4

# Action index -> (dx, dy) movement mapping
const ACTION_MOVEMENTS = Dict(
    ACTION_UP => (0, 1),
    ACTION_RIGHT => (1, 0),
    ACTION_DOWN => (0, -1),
    ACTION_LEFT => (-1, 0)
)

# Action index -> name mapping
const ACTION_NAMES = Dict(
    ACTION_UP => "up",
    ACTION_RIGHT => "right",
    ACTION_DOWN => "down",
    ACTION_LEFT => "left"
)

"""
    action_to_one_hot(action_idx::Int)

Convert action index to one-hot vector of length 4.
"""
function action_to_one_hot(action_idx::Int)
    v = zeros(4)
    v[action_idx] = 1.0
    return v
end

export ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT
export ACTION_MOVEMENTS, ACTION_NAMES, action_to_one_hot
