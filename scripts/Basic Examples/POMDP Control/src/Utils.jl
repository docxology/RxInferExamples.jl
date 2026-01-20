# Utility functions for POMDP Control
# See: docs/agent_architecture.md for logging documentation

using Dates

# Log file path from config
const POMDP_LOGFILE = joinpath(@__DIR__, "..", logging_params.log_file)

"""
    rxlog(level, msg)

Log a timestamped message to the POMDP log file and optionally console.
Controlled by [logging] section of config.toml

See: docs/agent_architecture.md#logging
"""
function rxlog(level::AbstractString, msg)
    ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    log_line = "[$ts] $(uppercase(level)) - $msg"
    
    # Console output based on log level
    if level == "debug" && logging_params.level != "debug"
        # Skip debug messages unless debug mode
    elseif level == "error"
        @error log_line
    elseif level == "warn"
        @warn log_line
    end
    
    # File output if enabled
    if logging_params.log_to_file
        open(POMDP_LOGFILE, "a") do io
            println(io, log_line)
        end
    end
end

"""
    grid_location_to_index(pos::Tuple{Int, Int}) -> Int

Convert a (x, y) grid position to a 1-indexed state index.

See: docs/environment_dynamics.md#state-representation
"""
function grid_location_to_index(pos::Tuple{Int,Int})
    return (pos[2] - 1) * grid_params.grid_size + pos[1]
end

"""
    index_to_grid_location(index::Int) -> Tuple{Int, Int}

Convert a 1-indexed state index to a (x, y) grid position.

See: docs/environment_dynamics.md#state-representation
"""
function index_to_grid_location(index::Int)
    x = ((index - 1) % grid_params.grid_size) + 1
    y = ((index - 1) ÷ grid_params.grid_size) + 1
    return (x, y)
end

"""
    index_to_one_hot(index::Int, n::Int) -> Vector{Float64}

Generate a one-hot vector of length n with 1.0 at the given index.
Default length is grid_size^2 (number of states).

See: docs/model_specification.md#one-hot-encoding
"""
function index_to_one_hot(index::Int, n::Int=grid_params.grid_size^2)
    return [i == index ? 1.0 : 0.0 for i in 1:n]
end

export rxlog
export grid_location_to_index, index_to_grid_location
export index_to_one_hot
