# POMDP Control

Active Inference POMDP agent navigating a WindyGridWorld using RxInfer.jl.

## Quick Start

```bash
# Run full analysis with visualization
julia --project=. run_full_analysis.jl

# Run parameter sweep
julia --project=. run_sweep.jl

# Headless verification
julia --project=. run_headless.jl
```

## Overview

This example demonstrates **POMDP (Partially Observable Markov Decision Process) control** via **Active Inference**, where planning is performed through probabilistic inference.

### The Task

An agent navigates a 5×5 grid from start `(1,1)` to goal `(4,3)`, learning to compensate for wind that pushes it upward in columns 2-4.

```
Y
5 │ · · ↑ ↑ ↑ ·     Wind in columns 2,3,4
4 │ · · ↑ ↑ ↑ ·     pushes agent upward
3 │ · · ↑ ★ ↑ ·     ★ = Goal (4,3)
2 │ · · ↑ ↑ ↑ ·
1 │ ○ · ↑ ↑ ↑ ·     ○ = Start (1,1)
  └───────────────
    1 2 3 4 5  X
```

## Mathematical Framework

### Generative Model

The agent maintains a generative model:

$$P(o_{1:T}, s_{0:T}, u_{1:T}) = P(s_0) \prod_{t=1}^{T} P(s_t | s_{t-1}, u_t) P(o_t | s_t)$$

Where:
- $s_t \in \{1, ..., 25\}$ — Hidden state (grid position)
- $o_t$ — Observation (perceived position)
- $u_t \in \{\text{up}, \text{right}, \text{down}, \text{left}\}$ — Action

### Observation Model (A Matrix)

$$A_{ij} = P(o_i | s_j)$$

- Prior: `DirichletCollection(I + 0.1)` — Weakly peaked on diagonal
- Learned online from observations

### Transition Model (B Matrix)

$$B_{ijk} = P(s'_i | s_j, u_k)$$

- Prior: `DirichletCollection(ones(25, 25, 4))` — Uniform
- Learns wind dynamics through experience

### Expected Free Energy

Actions selected to minimize Expected Free Energy:

$$G(\pi) = \underbrace{\mathbb{E}[\log Q(s) - \log P(s)]}_{\text{Epistemic}} + \underbrace{\mathbb{E}[-\log P(o)]}_{\text{Pragmatic}}$$

The goal prior $P(s_T)$ centered on `(4,3)` drives pragmatic behavior.

## Configuration

All parameters in [`config.toml`](config.toml):

```toml
[experiment]
n_experiments = 100      # Experiments per replicate
planning_horizon = 4     # Steps per experiment
n_iterations = 20        # VMP iterations
n_replicates = 1         # Independent replicates

[priors]
A_diagonal_bias = 0.1    # Observation model prior
B_concentration = 1.0    # Transition model prior
```

See [`docs/configuration.md`](docs/configuration.md) for full reference.

## Project Structure

```
├── config.toml          # All parameters
├── run_full_analysis.jl # Full experiment with stats
├── run_sweep.jl         # Parameter sweep
├── run_headless.jl      # CI verification
├── src/
│   ├── Config.jl        # TOML loader
│   ├── Constants.jl     # Action definitions
│   ├── Utils.jl         # Logging, conversions
│   ├── World.jl         # WindyGridWorld
│   ├── Model.jl         # RxInfer @model
│   ├── Agent.jl         # PODMPAgent
│   └── visualization/   # Plots, animations
├── docs/                # Documentation
└── output/              # Generated outputs
```

## Typical Results

With default configuration (100 experiments, T=4):

| Metric | Value |
|--------|-------|
| Final success rate | 82% |
| Convergence (>50%) | ~37 experiments |
| Mean steps to goal | 3.2 |

With extended training (200 experiments):

| Metric | Value |
|--------|-------|
| Final success rate | 91% |

## Documentation

- [`docs/README.md`](docs/README.md) — Documentation index
- [`docs/mathematical_foundations.md`](docs/mathematical_foundations.md) — Theory
- [`docs/model_specification.md`](docs/model_specification.md) — RxInfer model
- [`docs/configuration.md`](docs/configuration.md) — Parameter reference

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Da Costa, L. et al. (2020). Active inference on discrete state-spaces
3. [RxInfer.jl Documentation](https://rxinfer.ml/)
