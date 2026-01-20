# Configuration Loader for POMDP Control
# Loads parameters from config.toml
# See: docs/configuration.md for parameter descriptions

using TOML

const CONFIG_FILE = joinpath(@__DIR__, "..", "config.toml")

"""
    load_config(config_path=CONFIG_FILE)

Load configuration from TOML file.
Returns Dict with all configuration sections.

See also: [`docs/configuration.md`](@ref)
"""
function load_config(config_path::String=CONFIG_FILE)
    if !isfile(config_path)
        @warn "Config file not found at $config_path, using defaults"
        return get_default_config()
    end
    
    config = TOML.parsefile(config_path)
    # Note: Can't use rxlog here as Utils.jl loads after Config.jl
    println("[Config] Loaded configuration from $config_path")
    return config
end

"""
    get_default_config()

Return default configuration if config.toml not found.
"""
function get_default_config()
    return Dict(
        "environment" => Dict(
            "grid_size" => 5,
            "wind" => [0, 1, 1, 1, 0],
            "start_x" => 1, "start_y" => 1,
            "goal_x" => 4, "goal_y" => 3
        ),
        "experiment" => Dict(
            "n_experiments" => 100,
            "planning_horizon" => 4,
            "n_iterations" => 20,
            "n_replicates" => 1,
            "random_seed" => 0
        ),
        "priors" => Dict(
            "A_diagonal_bias" => 0.1,
            "B_concentration" => 1.0
        ),
        "logging" => Dict(
            "level" => "info",
            "log_to_file" => true,
            "log_file" => "pomdp.log",
            "show_progress" => true
        ),
        "verification" => Dict(
            "min_success_rate" => 0.5,
            "verification_experiments" => 50
        ),
        "visualization" => Dict(
            "enabled" => true,
            "generate_plots" => true,
            "generate_animations" => true,
            "export_data" => true,
            "animation_fps" => 2,
            "figure_dpi" => 150
        )
    )
end

# Load config on module initialization
const CONFIG = load_config()

"""
    get_config(section::String, key::String, default=nothing)

Get a configuration value with optional default.

# Example
```julia
n_exp = get_config("experiment", "n_experiments", 100)
```
"""
function get_config(section::String, key::String, default=nothing)
    if haskey(CONFIG, section) && haskey(CONFIG[section], key)
        return CONFIG[section][key]
    end
    return default
end

"""
    get_config_section(section::String)

Get entire configuration section as Dict.
"""
function get_config_section(section::String)
    return get(CONFIG, section, Dict())
end

# Build typed configuration structs from TOML

"""
Grid world configuration parameters.
Loaded from [environment] section of config.toml
"""
const grid_params = (
    grid_size = get_config("environment", "grid_size", 5),
    wind = Tuple(get_config("environment", "wind", [0, 1, 1, 1, 0])),
    goal = (get_config("environment", "goal_x", 4), 
            get_config("environment", "goal_y", 3)),
    start = (get_config("environment", "start_x", 1),
             get_config("environment", "start_y", 1))
)

"""
Experiment configuration parameters.
Loaded from [experiment] section of config.toml
"""
const experiment_params = (
    n_experiments = get_config("experiment", "n_experiments", 100),
    planning_horizon = get_config("experiment", "planning_horizon", 4),
    n_iterations = get_config("experiment", "n_iterations", 20),
    n_replicates = get_config("experiment", "n_replicates", 1),
    random_seed = get_config("experiment", "random_seed", 0),
    stochastic_actions = get_config("experiment", "stochastic_actions", false),
    exploration_temperature = get_config("experiment", "exploration_temperature", 1.0)
)

"""
Prior distribution parameters.
Loaded from [priors] section of config.toml
"""
const prior_params = (
    A_diagonal_bias = get_config("priors", "A_diagonal_bias", 0.1),
    B_concentration = get_config("priors", "B_concentration", 1.0)
)

"""
Logging configuration.
Loaded from [logging] section of config.toml
"""
const logging_params = (
    level = get_config("logging", "level", "info"),
    log_to_file = get_config("logging", "log_to_file", true),
    log_file = get_config("logging", "log_file", "pomdp.log"),
    show_progress = get_config("logging", "show_progress", true)
)

"""
Verification configuration.
Loaded from [verification] section of config.toml
"""
const verification_params = (
    min_success_rate = get_config("verification", "min_success_rate", 0.5),
    verification_experiments = get_config("verification", "verification_experiments", 50)
)

"""
Visualization configuration.
Loaded from [visualization] section of config.toml
"""
const viz_params = (
    enabled = get_config("visualization", "enabled", true),
    generate_plots = get_config("visualization", "generate_plots", true),
    generate_animations = get_config("visualization", "generate_animations", true),
    export_data = get_config("visualization", "export_data", true),
    animation_fps = get_config("visualization", "animation_fps", 2),
    figure_dpi = get_config("visualization", "figure_dpi", 150)
)

export CONFIG, load_config, get_config, get_config_section
export grid_params, experiment_params, prior_params
export logging_params, verification_params, viz_params
