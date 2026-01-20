# AGENTS.md - AI Development Guide

Instructions for AI agents working on this POMDP Control example.

## Architecture Overview

```
PODMPControlApp (main module)
├── Config.jl      → Loads config.toml, exports typed structs
├── Constants.jl   → ACTION_UP/RIGHT/DOWN/LEFT, movements
├── Utils.jl       → rxlog(), grid↔index conversions
├── World.jl       → WindyGridWorld + RxEnvironments
├── Model.jl       → @model pomdp_model, @constraints
├── Agent.jl       → PODMPAgent struct, step!, run_experiment!
└── visualization/
    ├── Visualization.jl  → Module index
    ├── MatrixPlots.jl    → A/B matrix heatmaps
    ├── BeliefPlots.jl    → State beliefs, policies
    ├── TrajectoryPlots.jl → Paths, action sequences
    ├── DiagnosticsPlots.jl → Learning curves
    ├── Animations.jl     → GIF generation
    └── DataExport.jl     → JSON export
```

## Key Patterns

### Adding New Parameters

1. Add to `config.toml` in appropriate section
2. Add to `Config.jl`:
   - Add to `get_default_config()`
   - Add to appropriate `*_params` struct
3. Use via `*_params.new_param` in code

### Adding New Visualizations

1. Create function in appropriate `*Plots.jl`
2. Follow pattern:
```julia
function plot_new_thing(agent; save_path=nothing)
    p = plot(...)
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved to $save_path")
    end
    return p
end
export plot_new_thing
```
3. Call from `run_full_analysis.jl` or `run_with_viz.jl`

### Modifying the Model

The RxInfer model is in `Model.jl`:
```julia
@model function pomdp_model(p_A, p_B, p_goal, p_control, ...)
    A ~ p_A
    B ~ p_B
    # ... state transitions and observations
end
```

Important: Changes to model structure require corresponding changes to inference call in `Agent.jl`.

## Testing Changes

```bash
# Quick verification
julia --project=. run_headless.jl

# Full test with visualization
julia --project=. run_full_analysis.jl

# Parameter sweep
julia --project=. run_sweep.jl
```

## Common Modifications

### Change Grid Size
Edit `config.toml` → `[environment]` → `grid_size`
(Also update `wind` array length to match)

### Change Wind Pattern
Edit `config.toml` → `[environment]` → `wind`
Array of integers, one per column

### Add New Action
1. Add constant in `Constants.jl`
2. Add to `ACTION_MOVEMENTS` and `ACTION_NAMES`
3. Update `action_to_one_hot()` to length 5
4. Update model priors for 5 actions

## File Dependencies

```
Config.jl (first - no deps)
    ↓
Constants.jl (uses Config)
    ↓
Utils.jl (uses Config)
    ↓
World.jl (uses Config, Constants, Utils)
    ↓
Model.jl (uses Config, Utils)
    ↓
Agent.jl (uses all above)
    ↓
visualization/* (uses all above)
```

## Debugging Tips

1. **Module load fails**: Check import order in `PODMPControlApp.jl`
2. **Config not loading**: Check `config.toml` syntax with `TOML.parsefile()`
3. **Inference errors**: Check data dimensions match model expectations
4. **Plot errors**: Ensure Plots.jl backend is set (`ENV["GKSwstype"]="100"` for headless)
