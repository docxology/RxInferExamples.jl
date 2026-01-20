# POMDP Control Documentation

Comprehensive documentation for the POMDP Control example demonstrating Active Inference in a WindyGridWorld environment using RxInfer.jl.

**Source Code**: [`src/`](../src/)  
**Configuration**: [`config.toml`](../config.toml)  
**Entry Points**: [`run.jl`](../run.jl) | [`run_headless.jl`](../run_headless.jl) | [`run_with_viz.jl`](../run_with_viz.jl)

## Contents

| Document | Description | Related Code |
|----------|-------------|--------------|
| [configuration.md](configuration.md) | config.toml parameter reference | [`src/Config.jl`](../src/Config.jl) |
| [mathematical_foundations.md](mathematical_foundations.md) | POMDP formalism, Active Inference, EFE | [`src/Model.jl`](../src/Model.jl) |
| [model_specification.md](model_specification.md) | RxInfer model, priors, constraints | [`src/Model.jl`](../src/Model.jl) |
| [inference_procedure.md](inference_procedure.md) | Message passing, belief updates | [`src/Agent.jl`](../src/Agent.jl) |
| [environment_dynamics.md](environment_dynamics.md) | WindyGridWorld mechanics | [`src/World.jl`](../src/World.jl) |
| [agent_architecture.md](agent_architecture.md) | Agent design, control loop | [`src/Agent.jl`](../src/Agent.jl) |
| [visualization_guide.md](visualization_guide.md) | Plots, animations, export | [`src/visualization/`](../src/visualization/) |

## Quick Start

```bash
# Edit configuration
vim config.toml

# Run interactive experiments
julia --project=. run.jl

# Run with visualization output
julia --project=. run_with_viz.jl

# Run headless verification
julia --project=. run_headless.jl
```

## Key Concepts

This example implements **POMDP-based Active Inference** where an agent:

1. **Perceives**: Receives observations of its grid position
2. **Believes**: Maintains posterior beliefs over hidden states
3. **Plans**: Uses Expected Free Energy (EFE) to select actions
4. **Learns**: Updates A (observation) and B (transition) matrices online

### Key Equations

**State Belief Update** ([inference_procedure.md](inference_procedure.md)):
$$P(s_t | o_{1:t}, u_{1:t-1}) \propto P(o_t | s_t) \sum_{s_{t-1}} P(s_t | s_{t-1}, u_{t-1}) P(s_{t-1})$$

**Expected Free Energy** ([mathematical_foundations.md](mathematical_foundations.md)):
$$G(\pi) = \sum_{\tau} \mathbb{E}_{Q} \left[ \log Q(s_\tau | \pi) - \log P(o_\tau, s_\tau | \pi) \right]$$

## Typical Results

After 50 experiments with online learning:
- **0% → 64% success rate** (agent learns transition dynamics with wind)
- Convergence typically occurs around experiment 30-40
- See [visualization_guide.md](visualization_guide.md) for output examples
