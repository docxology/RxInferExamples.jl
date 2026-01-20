# Configuration Guide

## Overview

All POMDP Control parameters are configurable via `config.toml` in the project root.

**File**: [`config.toml`](../config.toml)  
**Loader**: [`src/Config.jl`](../src/Config.jl)

## Configuration Sections

### [environment]

Grid world settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | Int | 5 | Grid dimensions (NxN) |
| `wind` | Array[Int] | [0,1,1,1,0] | Wind strength per column |
| `start_x`, `start_y` | Int | 1, 1 | Agent start position |
| `goal_x`, `goal_y` | Int | 4, 3 | Goal position |

**See**: [environment_dynamics.md](environment_dynamics.md)

### [experiment]

Experiment execution settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_experiments` | Int | 100 | Experiments per sequence |
| `planning_horizon` | Int | 4 | Steps per experiment |
| `n_iterations` | Int | 20 | VMP iterations per inference |
| `n_replicates` | Int | 1 | Independent replicate sequences |
| `random_seed` | Int | 0 | RNG seed (0 = system time) |

**See**: [inference_procedure.md](inference_procedure.md)

### [priors]

Prior distribution settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_diagonal_bias` | Float | 0.1 | Added to A matrix diagonal |
| `B_concentration` | Float | 1.0 | B matrix Dirichlet concentration |

**See**: [mathematical_foundations.md](mathematical_foundations.md#a-matrix-observation-likelihood)

### [logging]

Logging configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | String | "info" | Log level: debug/info/warn/error |
| `log_to_file` | Bool | true | Write logs to file |
| `log_file` | String | "pomdp.log" | Log file path |
| `show_progress` | Bool | true | Show progress bar |

**See**: [agent_architecture.md](agent_architecture.md#logging)

### [verification]

Headless verification settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_success_rate` | Float | 0.5 | Pass threshold (0-1) |
| `verification_experiments` | Int | 50 | Experiments for verification |

### [visualization]

Output generation settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | Bool | true | Enable visualization |
| `generate_plots` | Bool | true | Generate static plots |
| `generate_animations` | Bool | true | Generate GIF animations |
| `export_data` | Bool | true | Export JSON data |
| `animation_fps` | Int | 2 | Animation frame rate |
| `figure_dpi` | Int | 150 | Plot resolution |

**See**: [visualization_guide.md](visualization_guide.md)

## Usage in Code

Parameters are accessed via typed structs:

```julia
# Grid parameters
grid_params.grid_size    # 5
grid_params.goal         # (4, 3)
grid_params.wind         # (0, 1, 1, 1, 0)

# Experiment parameters
experiment_params.n_experiments    # 100
experiment_params.planning_horizon # 4
experiment_params.n_replicates     # 1

# Prior parameters
prior_params.A_diagonal_bias  # 0.1
prior_params.B_concentration  # 1.0

# Logging parameters
logging_params.level      # "info"
logging_params.log_to_file  # true

# Verification parameters
verification_params.min_success_rate  # 0.5

# Visualization parameters
viz_params.animation_fps  # 2
```

## Dynamic Access

For programmatic access:

```julia
# Get single value with default
n = get_config("experiment", "n_experiments", 100)

# Get entire section
env_config = get_config_section("environment")
```

## Example: Custom Configuration

```toml
# config.toml - Fast test run

[experiment]
n_experiments = 10
n_replicates = 3
planning_horizon = 4

[logging]
level = "debug"
show_progress = false

[visualization]
generate_animations = false  # Faster
```

## Validation

The config loader validates:
- File existence (falls back to defaults if missing)
- Type correctness (via TOML parsing)

Missing keys use defaults from `get_default_config()`.
