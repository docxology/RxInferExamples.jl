# Visualization Guide

## Overview

The visualization submodule in [`src/visualization/`](../src/visualization/) provides comprehensive plotting, animation, and data export capabilities.

## Module Structure

```
src/visualization/
├── Visualization.jl     # Module index, output management
├── MatrixPlots.jl       # A/B matrix heatmaps
├── BeliefPlots.jl       # State beliefs and policies
├── TrajectoryPlots.jl   # Agent trajectories
├── DiagnosticsPlots.jl  # Learning curves
├── Animations.jl        # GIF generation
└── DataExport.jl        # JSON export
```

## Output Directory

```
output/
├── animations/
│   ├── trajectory.gif
│   ├── belief_evolution.gif
│   └── learning_progress.gif
├── data/
│   ├── agent_state_*.json
│   ├── learning_history_*.json
│   ├── trajectory_*.json
│   └── model_spec_*.json
└── figures/
    ├── matrices/
    ├── beliefs/
    ├── trajectories/
    └── diagnostics/
```

## Matrix Visualizations

### Observation Matrix (A)

```julia
plot_A_matrix(agent, save_path="output/figures/matrices/A_matrix.png")
```

Heatmap of P(observation | state). Diagonal structure indicates agent learned that observations reveal true state.

### Transition Matrices (B)

```julia
# Single action
plot_B_matrix(agent, ACTION_RIGHT, save_path="B_right.png")

# All 4 actions in grid
plot_all_B_matrices(agent, save_dir="output/figures/matrices/")

# Arrow field showing expected transitions
plot_B_matrix_as_arrows(agent, ACTION_UP)
```

### Grid Overlays

```julia
# A matrix diagonal as grid heatmap
plot_A_matrix_as_grid(agent)
```

## Belief Visualizations

### State Belief

```julia
plot_state_belief(agent)
```

Shows probability distribution over grid positions with start/goal markers.

### Policy

```julia
plot_policy(agent)
```

Bar chart of action probabilities for each planning step.

### Goal Prior

```julia
plot_goal_prior(agent)
```

Visualizes the categorical goal distribution.

### Entropy Tracking

```julia
beliefs = [agent.p_s, ...]  # collected over time
plot_belief_entropy(beliefs)
```

## Trajectory Visualizations

### Recording

```julia
recorder = TrajectoryRecorder()

for t in 1:T
    action_idx = mode(first(agent.current_policy))
    step!(agent, T - t)
    record_step!(recorder, agent, action_idx, obs, time())
end
```

### Plotting

```julia
# Path on grid
plot_trajectory(recorder.positions)

# Action sequence bar chart
plot_action_sequence(recorder.actions)

# Combined summary
plot_experiment_summary(recorder)
```

## Diagnostics

### Learning History

```julia
history = LearningHistory()

for exp in 1:n_experiments
    success = run_experiment!(agent, T)
    record_experiment!(history, exp, success, steps, agent)
end
```

### Plots

```julia
# Learning curve
plot_learning_curve(history)

# Matrix entropy over experiments
plot_matrix_entropy_evolution(history)

# Steps distribution
plot_steps_histogram(history)

# Success/failure by experiment
plot_success_by_experiment(history)

# Combined dashboard (2×2 grid)
plot_diagnostics_dashboard(history)
```

## Animations

### Trajectory Animation

```julia
animate_trajectory(recorder.positions, 
    fps=2, 
    save_path="output/animations/trajectory.gif")
```

Shows agent moving through grid step by step.

### Belief Evolution

```julia
animate_belief_evolution(recorder.beliefs, recorder.positions,
    fps=2,
    save_path="output/animations/belief_evolution.gif")
```

Side-by-side view of belief heatmap and trajectory.

### Learning Progress

```julia
animate_learning_progress(history,
    fps=5,
    save_path="output/animations/learning_progress.gif")
```

Animated learning curve building up over experiments.

### Matrix Evolution

```julia
animate_matrix_evolution(A_history, B_history,
    fps=3,
    save_path="output/animations/matrix_evolution.gif")
```

Shows A and B matrices sharpening during learning.

## Data Export

### Agent State

```julia
export_agent_state(agent, save_path="agent_state.json")
```

Exports:
- Timestamp
- Grid parameters
- Step count
- A/B matrices (4 decimal precision)
- State belief
- Current policy

### Learning History

```julia
export_learning_history(history, save_path="history.json")
```

Exports:
- Per-experiment: success, steps, entropy, cumulative rate
- Summary statistics

### Trajectory

```julia
export_trajectory(recorder, save_path="trajectory.json")
```

Exports:
- Per-step: position, action, observation, timing

### Model Specification

```julia
export_model_specification(save_path="model_spec.json")
```

Exports:
- Environment config
- Inference parameters
- Prior descriptions

### All at Once

```julia
export_all_diagnostics(agent, history, recorder, output_dir="output/")
```

## Usage Examples

### Basic Visualization

```julia
include("src/PODMPControlApp.jl")
using .PODMPControlApp

# Setup
env, rx_agent = create_environment()
agent = PODMPAgent(env, rx_agent)
ensure_output_dirs()

# Run experiments
for _ in 1:50
    run_experiment!(agent, 4)
end

# Visualize
plot_A_matrix(agent, save_path="output/figures/matrices/A.png")
plot_state_belief(agent, save_path="output/figures/beliefs/final.png")
```

### Full Visualization Run

```bash
julia --project=. run_with_viz.jl
```

Generates all plots, animations, and data exports automatically.
